from transformers import pipeline

print("kr_to_eng_model.py import됨")

hf_token = "your_huggingface_token"

# 처음 생성될 때 한 번만 로딩
ko2en_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en", token=hf_token)

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
