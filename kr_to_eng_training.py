from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
import numpy as np
import evaluate

#pip install sacrebleu
#just how to get sacrebleu score

"""
metric = evaluate.load("sacrebleu")
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
"""

model_checkpoint = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


#data setup

from datasets import load_dataset

data_files = {
    "train": "data/train.jsonl",   # 여기에 네가 만든 경로
    "test": "data/test.jsonl",
}


raw_datasets = load_dataset("json",  data_files=data_files)
#split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
#split_datasets["train"][1]["translation"]
split_datasets = raw_datasets




max_length = 128 #임의의 값 적당히 크게
def preprocess_function(examples):
    inputs = [ex["ko"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)




#fine tuning start
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) #skip special token = bos, pad 같은 것들 제거

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds] #pre.strip() 앞 뒤 붙은 공백 제거
    decoded_labels = [[label.strip()] for label in decoded_labels] #sacrebleu 입력 형식 맞추기

    result = metric.compute(predictions=decoded_preds, references=decoded_labels) #decoded_preds: 모델이 생성한 문장 리스트, decoded_labels: 각 문장마다 정답 리스트
    return {"bleu": result["score"]} #result는 dictionary

from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./sihyung-finetuned-ko-to-en",
    evaluation_strategy="epoch", #cpu 느리면 no
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, #32
    per_device_eval_batch_size=8, #64
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True, #평가에서 model.generate() 사용
    fp16=False,
    push_to_hub=False,
    no_cuda=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.evaluate(max_length=max_length)

trainer.train()
trainer.save_model() #arg에 나와 있는 폴더에 저장

#trainer.train(resume_from_checkpoint=True)
##trainer.train(resume_from_checkpoint="./sihyung-finetuned-ko-to-en/checkpoint-1000")
#멈췄던 training 다시 시작

"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("./sihyung-finetuned-ko-to-en")
model = AutoModelForSeq2SeqLM.from_pretrained("./sihyung-finetuned-ko-to-en")

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

args = Seq2SeqTrainingArguments(
    output_dir="./sihyung-finetuned-ko-to-en-v2",  # 새 output 폴더
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    no_cuda=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()   # ← 여기서부터는 "새 학습"이라고 보면 됨
trainer.save_model()
"""

trainer.evaluate(max_length=max_length)


