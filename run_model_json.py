from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import json
from pathlib import Path

MODEL_DIR = "./out-tf"  # 학습된 summarization 모델 폴더
PREFIX = "summarize: "

# 입력 / 출력 파일 경로
SRC_PATH = Path("./ex_lol_eng.json")          # 위에서 보여준 입력 JSON
DST_PATH = Path("./ex_lol_eng_summ.json")     # 요약 결과 JSON

# ----- 모델 로드 -----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)


def summarize(texts):
    # texts: str 또는 list[str]
    if isinstance(texts, str):
        texts = [texts]
    inputs = [PREFIX + t for t in texts]
    enc = tokenizer(
        inputs,
        max_length=512,       # 필요에 맞게 조절
        truncation=True,
        padding=True,
        return_tensors="tf",
    )
    output_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=128,
        num_beams=4,
    )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def load_json_or_default(path: Path, default):
    """파일이 없거나 비어 있거나 JSON 파싱이 안 되면 default 리턴."""
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[경고] {path} 는 유효한 JSON이 아닙니다. 기본값으로 대체합니다.")
        return default


def main():
    # 1. 입력 JSON 로드
    # 형식: [ { "translation": { "ko": "...", "en": "..." } }, ... ]
    src_data = load_json_or_default(SRC_PATH, [])

    # 2. 기존 요약 결과 JSON 로드 (없으면 []로 시작)
    dst_data = load_json_or_default(DST_PATH, [])

    # 3. 이미 요약된 en 문장 set (중복 방지용)
    existing_en = set()
    for item in dst_data:
        try:
            existing_en.add(item["translation"]["en"])
        except (KeyError, TypeError):
            continue

    total = len(src_data)
    print(f"총 {total}개 항목 처리 예정")

    new_count = 0

    for i, item in enumerate(src_data, start=1):
        # item에서 en만 뽑기
        try:
            tr = item["translation"]
            en_text = str(tr["en"]).strip()
            ko_text = str(tr.get("ko", "")).strip()
        except (KeyError, TypeError):
            print(f"[{i}/{total}] translation/en 형식이 아님, 건너뜀: {repr(item)[:50]}")
            continue

        if not en_text:
            print(f"[{i}/{total}] en이 비어 있음, 건너뜀")
            continue

        if en_text in existing_en:
            print(f"[{i}/{total}] 이미 요약된 en 문장, 건너뜀: {en_text[:60]!r}")
            continue

        print(f"[{i}/{total}] 요약 중 (en): {en_text[:80]!r}")
        summary_en = summarize(en_text)[0]

        # 출력 JSON에 추가 (ko, en 그대로 유지 + summary_en 추가)
        dst_data.append({
            "translation": {
                "ko": ko_text,
                "en": en_text,
                "summary_en": summary_en
            }
        })
        existing_en.add(en_text)
        new_count += 1

    # 4. 결과 JSON 저장 (전체 리스트를 덮어쓰지만 내용은 누적 상태)
    with DST_PATH.open("w", encoding="utf-8") as f:
        json.dump(dst_data, f, ensure_ascii=False, indent=2)

    print("요약 결과 저장 완료:", DST_PATH)
    print(f"새로 추가된 항목 수: {new_count}")


if __name__ == "__main__":
    main()
