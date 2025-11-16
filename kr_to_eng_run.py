from kr_to_eng_model import translate_ko2en
import re

print("finish import")

# 원본 & 결과 파일 경로
src_path = "./example_text_rough.txt"
dst_path = "./example_text_eng.txt"

# 한글 포함 여부 체크용 정규식
KOREAN_RE = re.compile(r"[가-힣]")

def is_korean_line(line: str) -> bool:
    return bool(KOREAN_RE.search(line))

def main():
    # 1. 원본 파일 전체 줄 읽기
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []

    # 2. 한국어 들어있는 줄만 골라서 번역
    for i, line in enumerate(lines, start=1):
        if is_korean_line(line):
            ko = line.strip()
            if not ko:
                continue
            print(f"[{i}/{len(lines)}] 번역 중: {ko[:50]!r}")
            en = translate_ko2en(ko)
            translated_lines.append(en + "\n")

    # 3. 새 파일에 영어 번역만 저장
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)

    print("번역 결과 저장 완료:", dst_path)

if __name__ == "__main__":
    print("hi")
    main()
