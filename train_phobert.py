import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

DATASET_PATH = "/workspace/DoAn/data/OFFICIAL_LAYOUTLM_DATASET"
MODEL_NAME   = "vinai/phobert-base-v2"
OUTPUT_DIR   = "/workspace/phobert_avir_official"
BEST_DIR     = "/workspace/phobert_avir_official_best"
MAX_SEQ_LEN  = 256
BATCH_SIZE   = 16
EPOCHS       = 10
LR           = 3e-5

LABELS = [
    "O",
    "B-SELLER",    "I-SELLER",
    "B-ADDRESS",   "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP",
    "B-TOTAL_COST","I-TOTAL_COST",
    "B-ITEM_NAME", "I-ITEM_NAME",
    "B-ITEM_PRICE","I-ITEM_PRICE",
    "B-ITEM_QUANTITY","I-ITEM_QUANTITY",
    "B-ITEM_TOTAL","I-ITEM_TOTAL",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
seqeval   = evaluate.load("seqeval")

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        is_split_into_words=True,
    )
    all_labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids     = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        labels       = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(label_ids[word_id] if word_id < len(label_ids) else -100)
            else:
                labels.append(-100)
            prev_word_id = word_id
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds  = [[ID2LABEL[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    true_labels = [[ID2LABEL[l] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

print("Loading dataset...")
ds = load_from_disk(DATASET_PATH)
print(ds)

print("Tokenizing...")
tokenized_ds = ds.map(tokenize_and_align, batched=True, remove_columns=ds["train"].column_names)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["val"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

print("Training PhoBERT...")
trainer.train()
trainer.save_model(BEST_DIR)
print(f"Best model saved to {BEST_DIR}")

print("Evaluating on test set...")
results = trainer.evaluate(tokenized_ds["test"])
print("Test results:", results)
