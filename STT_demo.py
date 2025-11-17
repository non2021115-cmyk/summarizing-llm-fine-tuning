import os
import time
import json
from glob import glob
from datetime import datetime
from transformers import pipeline

model_name = "openai/whisper-tiny"

asr = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=10,
    device="cuda:0"
)

INPUT_DIR = "wav_inputs"
OUTPUT_DIR = "json_outputs"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

processed_files = set()

def is_file_ready(path, checks=3, interval=0.5):
    last_size = -1
    for _ in range(checks):
        size = os.path.getsize(path)
        if size == last_size:
            return True
        last_size = size
        time.sleep(interval)
    return False

def transcribe_file(wav_path):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    json_path = os.path.join(OUTPUT_DIR, base + ".json")

    print(f"[INFO] 처리 중: {wav_path}")

    result = asr(
        wav_path,
        generate_kwargs={"task": "transcribe"}
    )
    text = result["text"]

    data = {
        "filename": os.path.basename(wav_path),
        "text": text,
        "model": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds")
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] JSON 저장 완료: {json_path}")
    return json_path

def main_loop(poll_interval=1.0):
    print(f"[START] 폴더 감시 시작: {os.path.abspath(INPUT_DIR)}")
    while True:
        wav_files = sorted(glob(os.path.join(INPUT_DIR, "*.wav")))

        for wav_path in wav_files:
            name = os.path.basename(wav_path)
            if name in processed_files:
                continue

            try:
                if not is_file_ready(wav_path):
                    continue
            except FileNotFoundError:
                continue

            try:
                transcribe_file(wav_path)
                processed_files.add(name)
            except Exception as e:
                print(f"[ERROR] {wav_path} 처리 중 오류: {e}")

        time.sleep(poll_interval)

if __name__ == "__main__":
    main_loop()
