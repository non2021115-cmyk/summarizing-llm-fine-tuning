import asyncio
import json
import subprocess
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper  # pip install -U openai-whisper
import json
from kr_to_eng_run_local_json import main as translator_run
from run_model_json import main as sum_run

translator_run()
sum_run()

# en_summary.json 파일 읽기 (server.py와 같은 폴더에 있다고 가정)
with open("en_summary.json", "r", encoding="utf-8") as f: #en_summary.json은 자기 파일 이름으로 바꾸기
    EN_SUMMARY_DATA = json.load(f)  # 리스트 형태로 로드됨

EN_SUMMARY_INDEX = 0  # 현재 몇 번째 문장을 쓰는지

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
        # 변환 실패 (너무 짧은 chunk거나 포맷 문제일 수 있음)
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
# 5. 번역 + 요약 훅 (너가 구현해야 하는 부분)
# ======================
def run_translation_and_summary(mixed_text: str) -> Dict[str, str]:
    """
    mixed_text는 사실 지금은 안 써도 됨.
    en_summary.json 안에 저장된 내용을 순서대로 하나씩 꺼내서 쓰는 함수.
    """

    global EN_SUMMARY_INDEX, EN_SUMMARY_DATA

    if not EN_SUMMARY_DATA:
        # 파일이 비어있거나 문제 있을 때 대비
        return {
            "ko": mixed_text,
            "en": "[NO_DATA] " + mixed_text,
            "summary_en": "[NO_DATA] " + mixed_text[:100] + "..."
        }

    # 현재 인덱스에 해당하는 항목 가져오기
    item = EN_SUMMARY_DATA[EN_SUMMARY_INDEX % len(EN_SUMMARY_DATA)]
    EN_SUMMARY_INDEX += 1  # 다음 호출 때는 그다음 문장

    trans = item.get("translation", {})

    ko = trans.get("ko", mixed_text)
    en = trans.get("en", "")
    summary_en = trans.get("summary_en", "")

    return {
        "ko": ko,
        "en": en,
        "summary_en": summary_en,
    }
    # --------------------------------------------------------------


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

            # 번역 + 요약 (네 모델로 교체)
            trans_dict = run_translation_and_summary(stt_text)

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
