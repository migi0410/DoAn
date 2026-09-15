import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer
)
from PIL import Image
import numpy as np
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

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
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/workspace/FINAL_LAYOUTLM_DATASET")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--output_dir", type=str, default="/workspace/layoutlmv3-final-model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)

    def prepare_examples(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        words = examples['tokens']
        boxes = examples['bboxes']
        word_labels = examples['ner_tags']

        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return encoding

    print("Processing dataset (This might take a while if not cached)...")
    
    # We apply the processing mapping
    features = Features({
        'input_ids': Sequence(Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Sequence(Sequence(Value(dtype='int64'))),
        'labels': Sequence(Value(dtype='int64')),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224))
    })

    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        features=features
    )
    
    val_dataset = dataset["val"].map(
        prepare_examples,
        batched=True,
        batch_size=8,
        remove_columns=dataset["val"].column_names,
        features=features
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=10000, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True, # enable mixed precision for faster training
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
