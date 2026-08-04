import os
import json
import torch
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
import subprocess; import sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools', 'wheel', 'setuptools_scm']); subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seqeval', '--no-build-isolation']); subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'evaluate']); import evaluate
import numpy as np

def prepare_dataset(data_path, processor):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Split: 80% train, 20% eval (simplified)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    hf_dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(eval_data)
    })
    
    label_list = ["O"] + [f"{bio}-{lbl}" for lbl in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"] for bio in ["B", "I"]]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    def process_func(examples):
        # We process ONLY text and bboxes here to save RAM and speed up mapping massively.
        # We'll inject dummy pixel_values in a custom data collator instead.
        encoding = processor.tokenizer(
            text=examples["words"],
            boxes=examples["bboxes"],
            truncation=True,
            padding="max_length",
            max_length=512,
            is_split_into_words=True
        )
        
        # Tự động align nhãn (labels) vì processor.tokenizer không tự làm điều này
        labels = []
        for i, batch_tags in enumerate(examples["ner_tags"]):
            word_ids = encoding.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    tag = batch_tags[word_idx]
                    label_ids.append(label2id[tag])
            labels.append(label_ids)
        encoding["labels"] = labels
        
        return encoding

    print("Processing dataset (Tokenization & Bbox Alignment)...")
    encoded_dataset = hf_dataset.map(process_func, batched=True, remove_columns=hf_dataset["train"].column_names)
    return encoded_dataset, label_list, label2id, id2label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to layoutlm_dataset.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    args = parser.parse_args()
    
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False, use_fast=False)
    
    dataset, label_list, label2id, id2label = prepare_dataset(args.data_path, processor)
    
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=1000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none"
    )
    
    def custom_collator(features):
        batch = DefaultDataCollator()(features)
        # Inject dummy images here during batching, saving 60GB of RAM during mapping!
        batch_size = len(features)
        batch["pixel_values"] = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32)
        return batch

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=processor,
            data_collator=custom_collator,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=processor,
            data_collator=custom_collator,
            compute_metrics=compute_metrics,
        )
    
    print(f"Bắt đầu huấn luyện mô hình và lưu tại {args.output_dir}...")
    trainer.train()
    
    print("Lưu mô hình tốt nhất...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Hoàn tất!")
