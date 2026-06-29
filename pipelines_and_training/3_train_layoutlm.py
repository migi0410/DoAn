# 3_train_layoutlm.py
import os
import json
import argparse
import numpy as np
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import (
    LayoutLMTokenizerFast, 
    LayoutLMForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
import evaluate

# Setup labels mapping
LABELS = [
    "O",
    "B-SELLER", "I-SELLER",
    "B-ADDRESS", "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP",
    "B-TOTAL_COST", "I-TOTAL_COST",
    "B-OTHER", "I-OTHER"
]
label2id = {lbl: idx for idx, lbl in enumerate(LABELS)}
id2label = {idx: lbl for idx, lbl in enumerate(LABELS)}

def normalize_bbox(bbox, width, height):
    """Normalize bounding box to 0-1000 scale"""
    return [
        max(0, min(1000, int(1000 * (bbox[0] / width)))),
        max(0, min(1000, int(1000 * (bbox[1] / height)))),
        max(0, min(1000, int(1000 * (bbox[2] / width)))),
        max(0, min(1000, int(1000 * (bbox[3] / height))))
    ]

def split_box_into_words(text, box):
    """Approximate word-level bounding boxes by splitting the phrase."""
    words = text.split()
    if not words:
        return [], []
    
    x1, y1, x2, y2 = box
    total_len = max(1, sum(len(w) for w in words))
    
    word_boxes = []
    current_x = x1
    for w in words:
        w_width = (len(w) / total_len) * (x2 - x1)
        word_boxes.append([current_x, y1, current_x + w_width, y2])
        current_x += w_width + (x2 - x1)*0.02 # add small space
        
    return words, word_boxes

def load_data_from_metadata(metadata_path, base_dir):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    all_words = []
    all_bboxes = []
    all_ner_tags = []
    
    for item in data:
        label_file = os.path.join(base_dir, item['label_path'])
        if not os.path.exists(label_file):
            continue
            
        with open(label_file, 'r', encoding='utf-8') as lf:
            label_data = json.load(lf)
            
        width = label_data.get('image_width', 600)
        height = label_data.get('image_height', 800)
        
        words = []
        bboxes = []
        ner_tags = []
        
        for ann in label_data.get('annotations', []):
            label = ann['label']
            text = ann['text']
            box = ann['box'] # [x1, y1, x2, y2]
            
            if label not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                label = "OTHER"
                
            w_list, box_list = split_box_into_words(text, box)
            
            for i, (w, b) in enumerate(zip(w_list, box_list)):
                words.append(w)
                norm_box = normalize_bbox(b, width, height)
                bboxes.append(norm_box)
                
                if i == 0:
                    ner_tags.append(label2id[f"B-{label}"])
                else:
                    ner_tags.append(label2id[f"I-{label}"])
                    
        if len(words) > 0:
            all_words.append(words)
            all_bboxes.append(bboxes)
            all_ner_tags.append(ner_tags)
            
    return {
        "id": list(range(len(all_words))),
        "words": all_words,
        "bboxes": all_bboxes,
        "ner_tags": all_ner_tags
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast train on just a few examples")
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlm-base-uncased")
    args = parser.parse_args()

    print("Loading datasets...")
    train_dict = load_data_from_metadata("synthetic_dataset/train/metadata.json", "synthetic_dataset")
    val_dict = load_data_from_metadata("synthetic_dataset/val/metadata.json", "synthetic_dataset")
    
    if args.smoke_test:
        print("Running in SMOKE TEST mode. Slicing data...")
        for key in train_dict:
            train_dict[key] = train_dict[key][:16]
        for key in val_dict:
            val_dict[key] = val_dict[key][:8]
            
    features = Features({
        "id": Value(dtype="int64"),
        "words": Sequence(Value(dtype="string")),
        "bboxes": Sequence(Sequence(Value(dtype="int64"))),
        "ner_tags": Sequence(ClassLabel(names=LABELS))
    })
    
    train_dataset = Dataset.from_dict(train_dict, features=features)
    val_dataset = Dataset.from_dict(val_dict, features=features)
    
    print("Tokenizing datasets...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(args.model_name)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["words"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            max_length=512
        )
        
        labels = []
        bboxes = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            bbox_list = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_list.append([0, 0, 0, 0])
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    bbox_list.append(examples["bboxes"][i][word_idx])
                else:
                    label_ids.append(-100)
                    bbox_list.append(examples["bboxes"][i][word_idx])
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            bboxes.append(bbox_list)
            
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs
        
    train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names)

    print("Loading model...")
    model = LayoutLMForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Metrics
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
        output_dir="./layoutlm-avir-kie",
        max_steps=50 if args.smoke_test else -1,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    data_collator = DefaultDataCollator()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    trainer.evaluate()
    
    # Save the final best model
    model_dir = "layoutlm-avir-kie-best"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    main()
