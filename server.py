import asyncio
import json
import subprocess
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper  # pip install -U openai-whisper

from kr_to_eng_run_local_json import main as translator_run
from run_model_json import main as sum_run
import os
from datetime import datetime

# ======================
# 1. FastAPI 기본 세팅
# ======================
app = FastAPI()

# CORS: 개발 편하게 로컬에서 테스트하기 위해 전체 허용 (원하면 나중에 조이기)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# 2. Whisper 모델 로드
# ======================
# 실시간이면 tiny/base 추천
WHISPER_MODEL_NAME = "tiny"  # "base", "small" 등으로 바꿔도 됨
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

# ======================
# 3. webm(opus) -> PCM float32 변환 함수 (FFmpeg 파이프 사용, 파일 저장 X)
# ======================
def webm_bytes_to_pcm_float32(
    webm_bytes: bytes,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """
    브라우저에서 온 webm/opus 바이트를 FFmpeg로
    16kHz mono PCM(int16) -> float32[-1, 1] numpy array로 변환.
    """

    # ffmpeg 가 PATH에 있어야 함 (cmd에서 `ffmpeg -version` 되는 상태)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            "pipe:0",           # stdin에서 입력
            "-f",
            "s16le",            # 16-bit PCM
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",                # mono
            "-ar",
            str(target_sample_rate),
            "pipe:1",           # stdout으로 출력
            "-loglevel",
            "quiet",            # 로그 안 찍기 (원하면 debug로 변경)
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    out, err = process.communicate(input=webm_bytes)

    if process.returncode != 0:
        # 변환 실패 (너무 짧거나 포맷 문제일 수 있음)
        # print("ffmpeg error:", err.decode("utf-8", errors="ignore"))
        return np.array([], dtype=np.float32)

    # int16 PCM -> float32 [-1.0, 1.0]
    if not out:
        return np.array([], dtype=np.float32)

    audio_int16 = np.frombuffer(out, np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return audio_float32


# ======================
# 4. Whisper로 chunk STT
# ======================
def run_whisper_stt_on_chunk(
    audio_pcm: np.ndarray,
    language: str = None,
) -> str:
    """
    16kHz mono float32 PCM 배열을 받아 Whisper STT 수행.
    language를 None으로 두면 자동 감지.
    """
    if audio_pcm.size == 0:
        return ""

    # Whisper에 바로 numpy array 넣기
    # note: fp16=False는 CPU에서 안정적으로 돌리기 위해
    result = whisper_model.transcribe(
        audio_pcm,
        language=language,  # 예: "ko" 강제도 가능
        task="transcribe",  # 번역은 여기서 안 함 (너 번역 모델에 맡김)
        fp16=False,
    )

    text = result.get("text", "").strip()
    return text


# ======================
# 5. 번역 + 요약 훅
# ======================
EN_SUMMARY_JSON_PATH = "en_summary.json"    # 번역+요약 결과를 저장하는 기존 파일
CHUNK_LOG_JSON_PATH = "chunk_log.json"      # 각 chunk별 결과를 쌓아둘 로그 파일

def run_translation_and_summary(mixed_text: str) -> Dict[str, str]:
    """
    1) translator_run() + sum_run()을 실행해서 en_summary.json을 최신 상태로 업데이트
    2) en_summary.json을 다시 열어서 마지막 항목을 꺼냄
    3) 그 안의 translation.ko / translation.en / translation.summary_en 을 리턴

    mixed_text는 네 파이프라인이 파일 기반으로 이미 설계되어 있다면
    여기서 직접 쓰지 않아도 될 수도 있음.
    (필요하면 mixed_text를 어떤 입력 파일에 써주는 로직을 추가해도 됨.)
    """

    # 1) 번역 + 요약 파이프라인 실행 → en_summary.json 갱신
    translator_run()
    sum_run()

    # 2) en_summary.json 읽기
    if not os.path.exists(EN_SUMMARY_JSON_PATH):
        # 파일이 없으면 fallback (필요에 따라 메시지 수정)
        return {
            "ko": mixed_text,
            "en": "[NO en_summary.json] " + mixed_text,
            "summary_en": "[NO en_summary.json] " + mixed_text[:120] + "..."
        }

    with open(EN_SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 파일이 깨져 있거나 쓰는 중일 때 대비
            return {
                "ko": mixed_text,
                "en": "[JSON ERROR] " + mixed_text,
                "summary_en": "[JSON ERROR] " + mixed_text[:120] + "..."
            }

    if not isinstance(data, list) or not data:
        # 리스트 아니거나 비어 있을 때 대비
        return {
            "ko": mixed_text,
            "en": "[EMPTY DATA] " + mixed_text,
            "summary_en": "[EMPTY DATA] " + mixed_text[:120] + "..."
        }

    # 3) 최신 chunk에 해당하는 마지막 요소 사용
    last_item = data[-1]
    trans = last_item.get("translation", {})

    ko = trans.get("ko", mixed_text)
    en = trans.get("en", "")
    summary_en = trans.get("summary_en", "")

    return {
        "ko": ko,
        "en": en,
        "summary_en": summary_en,
    }


# ======================
# 5-1. chunk 결과를 별도 JSON 로그로 저장
# ======================
def append_chunk_to_log(
    stt_text: str,
    trans_dict: Dict[str, str],
    log_path: str = CHUNK_LOG_JSON_PATH,
):
    """
    각 chunk마다 STT 결과 + 번역 + 요약을 하나의 entry로
    chunk_log.json 같은 파일에 계속 쌓아줌.

    구조 예시:
    [
      {
        "timestamp": "...",
        "stt": "...",
        "translation": {
          "ko": "...",
          "en": "...",
          "summary_en": "..."
        }
      },
      ...
    ]
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stt": stt_text,
        "translation": {
            "ko": trans_dict.get("ko", ""),
            "en": trans_dict.get("en", ""),
            "summary_en": trans_dict.get("summary_en", ""),
        },
    }

    # 기존 로그 읽기
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    # 새 entry 추가
    data.append(entry)

    # 다시 저장
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================
# 6. WebSocket 엔드포인트
# ======================
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """
    브라우저(WebSocket) <-> 서버 스트리밍 엔드포인트.

    브라우저:
      - MediaRecorder로 webm chunk를 ws.send(...)로 계속 전송.
    서버:
      - chunk 수신 -> ffmpeg로 PCM 변환 -> Whisper STT -> 번역/요약
      - JSON 배열 형태로 결과를 ws.send_text(...)로 프론트에 push.
      - 동시에 chunk_log.json에 결과를 하나씩 쌓아서 저장.
    """
    await ws.accept()
    print("WebSocket connected")

    try:
        while True:
            # 브라우저에서 binary(webm chunk) 받기
            chunk_bytes = await ws.receive_bytes()

            # webm -> PCM float32
            pcm = webm_bytes_to_pcm_float32(chunk_bytes, target_sample_rate=16000)
            if pcm.size == 0:
                # 너무 짧거나 변환 실패하면 그냥 스킵
                continue

            # Whisper STT (한국어+영어 혼합 텍스트)
            stt_text = run_whisper_stt_on_chunk(pcm)
            if not stt_text:
                continue

            # 번역 + 요약 (네 파이프라인)
            trans_dict = run_translation_and_summary(stt_text)

            # === 여기서 chunk_log.json에 저장 ===
            append_chunk_to_log(stt_text, trans_dict)

            # 프론트에서 기대하는 JSON 형태:
            # [
            #   {
            #     "translation": {
            #       "ko": "...",
            #       "en": "...",
            #       "summary_en": "..."
            #     }
            #   }
            # ]
            payload: List[Dict[str, Any]] = [
                {
                    "translation": {
                        "ko": trans_dict.get("ko", ""),
                        "en": trans_dict.get("en", ""),
                        "summary_en": trans_dict.get("summary_en", ""),
                    }
                }
            ]

            # ensure_ascii=False: 한글 깨지지 않게
            await ws.send_text(json.dumps(payload, ensure_ascii=False))

            # 너무 과도한 연산 방지용 살짝 쉰다 (옵션)
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("Error in websocket_transcribe:", e)
        try:
            await ws.close()
        except Exception:
            pass


# ======================
# 7. 서버 실행
# ======================
if __name__ == "__main__":
    # 개발용 실행
    # http://localhost:8000/docs (REST용), WebSocket은 ws://localhost:8000/ws/transcribe
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
