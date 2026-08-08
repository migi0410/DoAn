import os
import json
import argparse
import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import (
    RobertaTokenizerFast,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

LABELS = [
    "O",
    "B-SELLER", "I-SELLER",
    "B-ADDRESS", "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP",
    "B-TOTAL_COST", "I-TOTAL_COST",
    "B-ITEM_NAME", "I-ITEM_NAME",
    "B-ITEM_QTY", "I-ITEM_QTY",
    "B-ITEM_PRICE", "I-ITEM_PRICE",
    "B-ITEM_AMOUNT", "I-ITEM_AMOUNT",
    "B-OTHER", "I-OTHER"
]
label2id = {lbl: idx for idx, lbl in enumerate(LABELS)}
id2label = {idx: lbl for idx, lbl in enumerate(LABELS)}

def load_data_1d(metadata_path, base_dir):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    all_words = []
    all_ner_tags = []
    
    for item in data:
        label_file = os.path.join(base_dir, item['label_path'])
        if not os.path.exists(label_file):
            continue
            
        with open(label_file, 'r', encoding='utf-8') as lf:
            label_data = json.load(lf)
            
        words = []
        ner_tags = []
        
        for ann in label_data.get('annotations', []):
            label = ann['label']
            text = ann['text']
            
            label = str(ann['label']).upper()
            if label.startswith("ITEM_NAME"):
                label = "ITEM_NAME"
            elif label.startswith("ITEM_QTY"):
                label = "ITEM_QTY"
            elif label.startswith("ITEM_PRICE"):
                label = "ITEM_PRICE"
            elif label.startswith("ITEM_AMOUNT"):
                label = "ITEM_AMOUNT"
            elif label not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                label = "OTHER"
                
            w_list = text.split()
            for i, w in enumerate(w_list):
                words.append(w)
                if i == 0:
                    ner_tags.append(label2id[f"B-{label}"])
                else:
                    ner_tags.append(label2id[f"I-{label}"])
                    
        if len(words) > 0:
            all_words.append(words)
            all_ner_tags.append(ner_tags)
            
    return {
        "id": list(range(len(all_words))),
        "words": all_words,
        "ner_tags": all_ner_tags
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model-name", type=str, default="vinai/phobert-base-v2")
    args = parser.parse_args()

    print("Loading 1D datasets for PhoBERT...")
    train_dict = load_data_1d("synthetic_dataset/train/metadata.json", "synthetic_dataset")
    val_dict = load_data_1d("synthetic_dataset/val/metadata.json", "synthetic_dataset")
    
    if args.smoke_test:
        for key in train_dict: train_dict[key] = train_dict[key][:16]
        for key in val_dict: val_dict[key] = val_dict[key][:8]
            
    features = Features({
        "id": Value(dtype="int64"),
        "words": Sequence(Value(dtype="string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS))
    })
    
    train_dataset = Dataset.from_dict(train_dict, features=features)
    val_dataset = Dataset.from_dict(val_dict, features=features)
    
    print("Tokenizing datasets...")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name, add_prefix_space=True)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["words"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=256 # PhoBERT max length is 256
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
    train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names)

    print("Loading PhoBERT model...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [LABELS[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABELS[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    epochs = 2 if args.smoke_test else 10
    
    training_args = TrainingArguments(
        output_dir="./phobert-avir-kie",
        max_steps=50 if args.smoke_test else -1,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting PhoBERT training...")
    trainer.train()
    trainer.evaluate()
    
    model_dir = "./phobert-avir-kie-best"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"PhoBERT Model saved to {model_dir}")

if __name__ == "__main__":
    main()
