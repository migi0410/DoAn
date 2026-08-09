import os
import numpy as np
from PIL import Image
os.environ["WANDB_PROJECT"] = "avir-kie-vlm"
os.environ["WANDB_ENTITY"]  = "haminhdung0410-fpt-university"
from datasets import load_from_disk
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATASET_PATH = "/workspace/DoAn/data/OFFICIAL_LAYOUTLM_DATASET"
IMAGE_DIR    = "/workspace/FINAL_RUNPOD_DATASET/images"
MODEL_NAME   = "microsoft/layoutlmv3-base"
OUTPUT_DIR   = "/workspace/layoutlmv3_avir_official"
WANDB_PROJECT = "avir-kie-vlm"
WANDB_ENTITY  = "haminhdung0410-fpt-university"
BEST_DIR     = "/workspace/layoutlmv3_avir_official_best"
MAX_SEQ_LEN  = 512
BATCH_SIZE   = 8
EPOCHS       = 10
LR           = 2e-5

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

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
seqeval   = evaluate.load("seqeval")

def load_image(image_path):
    basename = os.path.basename(image_path)
    full_path = os.path.join(IMAGE_DIR, basename)
    if os.path.exists(full_path):
        return Image.open(full_path).convert("RGB")
    return Image.new("RGB", (800, 1000), color=(255, 255, 255))

def preprocess(examples):
    images   = [load_image(p) for p in examples["image_path"]]
    encodings = processor(
        images,
        examples["tokens"],
        boxes=examples["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )
    all_labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids     = encodings.word_ids(batch_index=i)
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
    encodings["labels"] = all_labels
    return encodings

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

print("Preprocessing (this may take a while due to image loading)...")
tokenized_ds = ds.map(
    preprocess, batched=True, batch_size=16,
    remove_columns=ds["train"].column_names,
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
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
    report_to="wandb",
    run_name="layoutlmv3-avir-official",
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["val"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

print("Training LayoutLMv3...")
import os as _os
_ckpt = OUTPUT_DIR
_resume = False
if _os.path.isdir(_ckpt):
    _checkpoints = [d for d in _os.listdir(_ckpt) if d.startswith("checkpoint")]
    if _checkpoints:
        _resume = True
        print(f"Resuming from checkpoint in {_ckpt}")
trainer.train(resume_from_checkpoint=_resume if _resume else None)
trainer.save_model(BEST_DIR)
print(f"Best model saved to {BEST_DIR}")

print("Evaluating on test set...")
results = trainer.evaluate(tokenized_ds["test"])
print("Test results:", results)
