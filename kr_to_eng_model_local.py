from transformers import pipeline

print("kr_to_eng_model.py import됨")

# 파인튜닝된 모델이 저장된 로컬 폴더 경로
MODEL_DIR = "./sihyung-finetuned-ko-to-en"

# 로컬 모델은 token 필요 없음
ko2en_pipe = pipeline(
    "translation",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,  # tokenizer도 같은 폴더에 있으니까 같이 지정
    # device=0  # GPU 쓰면 이렇게, CPU만 쓸 거면 생략해도 됨
)

def translate_ko2en(text: str) -> str:
    out = ko2en_pipe(text, max_length=512)[0]["translation_text"]
    return out

def main():
    while True:
        text = input(">")
        en_text = translate_ko2en(text)
        print("en_text:", en_text)

if __name__ == "__main__":
    main()
