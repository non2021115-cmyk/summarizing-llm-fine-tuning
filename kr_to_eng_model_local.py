import re
from transformers import pipeline

print("kr_to_eng_model_local.py import됨")

# 파인튜닝된 모델이 저장된 로컬 폴더 경로
MODEL_DIR = "./sihyung-finetuned-ko-to-en"

# 로컬 모델은 token 필요 없음
ko2en_pipe = pipeline(
    "translation",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,  # tokenizer도 같은 폴더에 있으니까 같이 지정
    # device=0  # GPU 쓰면 이렇게, CPU만 쓸 거면 생략해도 됨
)

#def translate_ko2en(text: str) -> str:
    #out = ko2en_pipe(text, max_length=512)[0]["translation_text"]
    #return out

def split_sentences_block(text: str):
    # JSON 안에 글자 그대로 "\\n" 이 있을 경우를 대비
    text = text.replace("\\n", "\n")

    # 실제 줄바꿈 기준으로 먼저 쪼개기
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    sentences = []
    for ln in lines:
        # . ? ! 뒤 공백 기준 + (간단한 한글 패턴은 사실 크게 의미는 없음)
        parts = re.split(
            r'(?<=[\.!?])\s+',
            ln
        )
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)

    return sentences  # ★ 이거 꼭 필요!

def translate_ko2en(text: str) -> str:
    # 긴 문단 → 여러 문장으로 쪼개기
    sentences = split_sentences_block(text)
    if not sentences:
        return ""

    # 배치 번역
    outputs = ko2en_pipe(
        sentences,
        max_length=256,  # 필요하면 조절
    )

    translated = [o["translation_text"] for o in outputs]
    # 번역 결과를 다시 한 덩어리로
    return " ".join(translated)

def main():
    while True:
        text = input(">")
        en_text = translate_ko2en(text)
        print("en_text:", en_text)

if __name__ == "__main__":
    main()
