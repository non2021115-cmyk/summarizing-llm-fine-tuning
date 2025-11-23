import json
import re
from pathlib import Path
from kr_to_eng_model_local import translate_ko2en

print("finish import")

# 파일 경로
SRC_PATH = Path("./test1_q.json")       # 입력 JSON
DST_PATH = Path("./ex_lol_eng.json")   # 출력 JSON

# 한글 포함 여부 체크용 정규식 (원한다면 그대로 사용)
KOREAN_RE = re.compile(r"[가-힣]")

def is_korean_text(text: str) -> bool:
    return bool(KOREAN_RE.search(text))


def load_json_or_default(path: Path, default):
    """파일이 있으면 JSON 로드, 없으면 default 리턴."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def main():
    # 1. 원본 JSON 로드
    # ex_lol.json 형식 예시:
    # ["문장1", "문장2", ...]
    # 또는 [{"ko": "문장1"}, {"ko": "문장2"}, ...] 도 처리 가능하게 만들자.
    src_data = load_json_or_default(SRC_PATH, [])

    # 2. 기존 번역 결과 JSON 로드 (없으면 빈 리스트)
    dst_data = load_json_or_default(DST_PATH, []) #json 안에 [] 이게 들어가 있어야 함

    # 3. 이미 번역된 한국어 문장 set (중복 방지용)
    existing_ko = set()
    for item in dst_data:
        try:
            existing_ko.add(item["translation"]["ko"])
        except (KeyError, TypeError):
            # 혹시 형식 안 맞는 데이터가 섞여 있어도 터지지 않게
            continue

    total = len(src_data)
    print(f"총 {total}개 항목 처리 예정")

    for i, item in enumerate(src_data, start=1):
        # item이 문자열 또는 {"ko": "..."} 둘 다 지원
        if isinstance(item, str):
            ko = item.strip()
        elif isinstance(item, dict) and "ko" in item:
            ko = str(item["ko"]).strip()
        else:
            # 형식이 다르면 스킵
            print(f"[{i}/{total}] 지원하지 않는 형식, 건너뜀: {repr(item)[:40]}")
            continue

        if not ko:
            print(f"[{i}/{total}] 빈 문자열, 건너뜀")
            continue

        if not is_korean_text(ko):
            print(f"[{i}/{total}] 한글 없음, 건너뜀: {ko[:30]!r}")
            continue

        if ko in existing_ko:
            print(f"[{i}/{total}] 이미 번역된 문장, 건너뜀: {ko[:30]!r}")
            continue

        print(f"[{i}/{total}] 번역 중: {ko[:50]!r}")
        en = translate_ko2en(ko)

        # 출력 JSON에 append (업데이트 형식)
        dst_data.append({
            "translation": {
                "ko": ko,
                "en": en
            }
        })
        existing_ko.add(ko)

    # 4. 결과를 다시 JSON으로 저장 (전체를 덮어쓰지만 내용은 append 된 상태)
    with DST_PATH.open("w", encoding="utf-8") as f:
        json.dump(dst_data, f, ensure_ascii=False, indent=2)

    print("번역 결과 저장 완료:", DST_PATH)


if __name__ == "__main__":
    print("hi")
    main()
