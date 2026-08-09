import json
import os
from datasets import load_from_disk, Dataset, DatasetDict

BASE = os.path.dirname(os.path.abspath(__file__))

def get_image_set(jsonl_path):
    imgs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            img = d.get("images", d.get("image", ""))
            if isinstance(img, list):
                img = img[0]
            basename = os.path.basename(img)
            imgs[basename] = d
    return imgs

print("Loading OFFICIAL_DATASET splits...")
train_imgs = get_image_set(os.path.join(BASE, "data/OFFICIAL_DATASET/train.jsonl"))
val_imgs   = get_image_set(os.path.join(BASE, "data/OFFICIAL_DATASET/val.jsonl"))
test_imgs  = get_image_set(os.path.join(BASE, "data/OFFICIAL_DATASET/test.jsonl"))
print(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

print("Loading LayoutLM dataset...")
ds = load_from_disk(os.path.join(BASE, "data/FINAL_LAYOUTLM_DATASET"))
all_rows = list(ds["train"]) + list(ds["val"])
print(f"  Total rows: {len(all_rows)}")

train_rows, val_rows, test_rows, skipped = [], [], [], 0
for row in all_rows:
    basename = os.path.basename(row["image_path"])
    if basename in train_imgs:
        train_rows.append(row)
    elif basename in val_imgs:
        val_rows.append(row)
    elif basename in test_imgs:
        test_rows.append(row)
    else:
        skipped += 1

print(f"\nRe-split result:")
print(f"  Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)} | Skipped: {skipped}")

out_ds = DatasetDict({
    "train": Dataset.from_list(train_rows),
    "val":   Dataset.from_list(val_rows),
    "test":  Dataset.from_list(test_rows),
})

out_path = os.path.join(BASE, "data/OFFICIAL_LAYOUTLM_DATASET")
out_ds.save_to_disk(out_path)
print(f"\nSaved to: {out_path}")

print("\nAlso creating BERT test JSON for benchmark...")
test_bert = []
for row in test_rows:
    basename = os.path.basename(row["image_path"])
    official = test_imgs.get(basename, {})
    gt = {}
    if official:
        try:
            resp = official["messages"][1]["content"] if isinstance(official.get("messages"), list) else ""
            start = resp.find("{")
            end = resp.rfind("}") + 1
            if start >= 0:
                gt = json.loads(resp[start:end])
        except Exception:
            pass
    test_bert.append({
        "file_name": f"images/{basename}",
        "tokens": row["tokens"],
        "bboxes": row["bboxes"],
        "ner_tags": row["ner_tags"],
        "gt_json": gt,
    })

bert_path = os.path.join(BASE, "data/test_official_bert.json")
with open(bert_path, "w", encoding="utf-8") as f:
    json.dump(test_bert, f, ensure_ascii=False, indent=2)
print(f"Saved BERT test: {bert_path} ({len(test_bert)} samples)")
