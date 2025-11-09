from huggingface_hub import notebook_login

notebook_login()

from datasets import load_dataset
from transformers import AutoTokenizer


checkpoint = "t5-small"
SAVE_DIR = "./ds_tok"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["original_text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    ds = load_dataset("json", data_files="./training_labeling.json", split="train", encoding="utf-8")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    print(ds["train"][0])

    tokenized_ds = ds.map(preprocess_function, batched=True)
    print(tokenized_ds)
    print(tokenized_ds["train"][0])
    tokenized_ds.save_to_disk(SAVE_DIR)

if __name__ == "__main__":
    main()
