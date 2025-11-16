from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

MODEL_DIR = "./out-tf"  # out-tf 폴더 경로 (PowerShell 경로 OK)
PREFIX = "summarize: "
sample_text = open("./example_text_eng.txt", encoding="utf-8").read()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)  # 폴더에서 직접 로드

def summarize(texts):
    # texts: str 또는 list[str]
    if isinstance(texts, str):
        texts = [texts]
    inputs = [PREFIX + t for t in texts]
    enc = tokenizer(inputs, max_length=1024, truncation=True, return_tensors="tf")
    # TF에서도 generate 사용 가능
    output_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=128,
        num_beams=4
    )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

if __name__ == "__main__":
    sample = sample_text
    print(summarize(sample)[0])

