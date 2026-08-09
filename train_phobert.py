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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, use_fast=False)
seqeval   = evaluate.load("seqeval")

BOS = tokenizer.bos_token_id or tokenizer.cls_token_id
EOS = tokenizer.eos_token_id or tokenizer.sep_token_id
PAD = tokenizer.pad_token_id

def tokenize_and_align(examples):
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for tokens, label_ids in zip(examples["tokens"], examples["ner_tags"]):
        encoded_words = [tokenizer.encode(w, add_special_tokens=False) for w in tokens]
        input_ids = [BOS]
        labels    = [-100]
        for word_tok, label in zip(encoded_words, label_ids):
            if not word_tok:
                continue
            input_ids.extend(word_tok)
            labels.extend([label] + [-100] * (len(word_tok) - 1))
        input_ids.append(EOS)
        labels.append(-100)

        input_ids = input_ids[:MAX_SEQ_LEN]
        labels    = labels[:MAX_SEQ_LEN]
        attn_mask = [1] * len(input_ids)
        pad_len   = MAX_SEQ_LEN - len(input_ids)
        input_ids += [PAD] * pad_len
        labels    += [-100] * pad_len
        attn_mask += [0] * pad_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attn_mask)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}

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
