# server_ws.py
import asyncio
import json
import os
import tempfile
import subprocess

import websockets
from transformers import pipeline

from kr_to_eng_model_local import translate_ko2en
from run_model_json import summarize

# 1) Whisper STT 파이프라인 (STT_demo.py와 같은 설정)
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    chunk_length_s=10,
    device="CPU",  # GPU 없으면 "cuda:0" 빼고 CPU로
)

WS_HOST = "0.0.0.0"
WS_PORT = 8000
WS_PATH = "/ws/transcribe"


async def handle_client(websocket, path):
    if path != WS_PATH:
        # 잘못된 경로로 접속하면 그냥 끊기
        await websocket.close()
        return

    print("[WS] client connected")

    try:
        async for message in websocket:
            # 브라우저에서 오는 건 binary (webm/opus)라고 가정
            if isinstance(message, str):
                # 혹시 문자열 ping 같은 게 오면 무시
                continue

            audio_bytes = message  # webm/opus 바이트

            # -- 1) webm → wav (16kHz, mono) 변환 (ffmpeg 필요) --
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
                f_in.write(audio_bytes)
                webm_path = f_in.name

            wav_path = webm_path.replace(".webm", ".wav")

            # ffmpeg 이용해 변환 (stdout/stderr는 버림)
            cmd = [
                "ffmpeg",
                "-y",
                "-i", webm_path,
                "-ac", "1",        # mono
                "-ar", "16000",    # 16kHz
                wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # webm 임시파일 삭제
            try:
                os.remove(webm_path)
            except FileNotFoundError:
                pass

            # -- 2) Whisper STT --
            try:
                result = asr(
                    wav_path,
                    generate_kwargs={"task": "transcribe"}
                )
            finally:
                # wav 파일도 바로 삭제 → 디스크 안 쌓이게
                try:
                    os.remove(wav_path)
                except FileNotFoundError:
                    pass

            text_ko_mixed = result.get("text", "").strip()
            if not text_ko_mixed:
                # 빈 결과면 그냥 스킵
                continue

            # -- 3) 한/영 혼합 → 영어 번역 --
            en_text = translate_ko2en(text_ko_mixed)

            # -- 4) 영어 요약 --
            # run_model_json.summarize()는 list[str]도 받으니까 그대로 사용 가능
            summary_text = summarize(en_text)[0]

            # -- 5) 프론트로 보낼 JSON 구조 --
            payload = {
                "translation": {
                    "ko": text_ko_mixed,
                    "en": en_text,
                    "summary_en": summary_text,
                }
            }

            await websocket.send(
                json.dumps(payload, ensure_ascii=False)
            )

    except websockets.exceptions.ConnectionClosed:
        print("[WS] client disconnected")

async def main():
    print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
