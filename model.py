# model_tf.py
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
import tensorflow as tf

CHECKPOINT = "t5-small"
DATA_DIR = "./ds_tok"
BATCH_SIZE = 8
EPOCHS = 4

def main():
    # 1) 로드
    tokenized_ds = load_from_disk(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

    # 2) 배치 콜레이터 (★ model 객체 전달, TF 텐서 반환)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        return_tensors="tf",
    )

    # 3) TF Dataset 변환
    train_tfds = tokenized_ds["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )
    eval_tfds = tokenized_ds["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    # 4) 학습
    # seq2seq 모델은 loss를 내부에서 계산하므로 기본 compile로 충분
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
    model.fit(train_tfds, validation_data=eval_tfds, epochs=EPOCHS)

    # 5) 저장(옵션)
    model.save_pretrained("./out-tf")
    tokenizer.save_pretrained("./out-tf")

if __name__ == "__main__":
    main()
