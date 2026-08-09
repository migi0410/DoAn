import json
import os
import shutil
from datasets import load_from_disk, Dataset, DatasetDict

OFFICIAL_DS  = "/workspace/DoAn/data/OFFICIAL_DATASET"
BBOX_FILE    = "/workspace/FINAL_BBOX_DATASET_V3.json"
LM_DATASET   = "/workspace/FINAL_LAYOUTLM_DATASET"
OUT_PATH     = "/workspace/DoAn/data/OFFICIAL_LAYOUTLM_DATASET"

LABEL2ID = {
    "O": 0,
    "B-SELLER": 1,    "I-SELLER": 2,
    "B-ADDRESS": 3,   "I-ADDRESS": 4,
    "B-TIMESTAMP": 5, "I-TIMESTAMP": 6,
    "B-TOTAL_COST": 7,"I-TOTAL_COST": 8,
    "B-ITEM_NAME": 9, "I-ITEM_NAME": 10,
    "B-ITEM_PRICE": 11,"I-ITEM_PRICE": 12,
    "B-ITEM_QUANTITY": 13,"I-ITEM_QUANTITY": 14,
    "B-ITEM_TOTAL": 15,"I-ITEM_TOTAL": 16,
}

def get_image_set(jsonl_path):
    imgs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            img = d.get("images", d.get("image", ""))
            if isinstance(img, list):
                img = img[0]
            imgs[os.path.basename(img)] = d
    return imgs

def annotation_to_row(file_name, annotations, img_w=800, img_h=1000):
    tokens, bboxes, tags = [], [], []
    for ann in annotations:
        label = ann.get("label", "O")
        text  = str(ann.get("text", ""))
        box   = ann.get("box", [0, 0, 100, 20])
        words = text.split()
        if not words:
            continue
        n      = len(words)
        w_each = (box[2] - box[0]) / max(n, 1)
        for i, word in enumerate(words):
            wb = [
                int(box[0] + i * w_each), box[1],
                int(box[0] + (i + 1) * w_each), box[3],
            ]
            lm_box = [
                min(1000, int(wb[0] * 1000 / img_w)),
                min(1000, int(wb[1] * 1000 / img_h)),
                min(1000, int(wb[2] * 1000 / img_w)),
                min(1000, int(wb[3] * 1000 / img_h)),
            ]
            tag = ("B-" if i == 0 else "I-") + label if label != "O" else "O"
            tokens.append(word)
            bboxes.append(lm_box)
            tags.append(LABEL2ID.get(tag, 0))
    return {
        "id": os.path.basename(file_name),
        "image_path": "images/" + os.path.basename(file_name),
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": tags,
    }

def build_split(img_dict, lm_all, bbox_lookup):
    rows, skipped = [], []
    for basename in img_dict:
        if basename in lm_all:
            rows.append(lm_all[basename])
        elif basename in bbox_lookup:
            rows.append(annotation_to_row(basename, bbox_lookup[basename]["annotations"]))
        else:
            skipped.append(basename)
    return rows, skipped

print("Loading OFFICIAL_DATASET splits...")
train_imgs = get_image_set(f"{OFFICIAL_DS}/train.jsonl")
val_imgs   = get_image_set(f"{OFFICIAL_DS}/val.jsonl")
test_imgs  = get_image_set(f"{OFFICIAL_DS}/test.jsonl")
print(f"  train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

print("Loading FINAL_LAYOUTLM_DATASET...")
ds = load_from_disk(LM_DATASET)
lm_all = {}
for split in ds:
    for p, row in zip(ds[split]["image_path"], ds[split]):
        lm_all[os.path.basename(p)] = row
print(f"  {len(lm_all)} images in LayoutLM dataset")

print("Loading FINAL_BBOX_DATASET_V3.json...")
with open(BBOX_FILE, "r", encoding="utf-8") as f:
    bbox_lookup = {os.path.basename(d["file_name"]): d for d in json.load(f)}
print(f"  {len(bbox_lookup)} images in BBOX dataset")

print("Building splits...")
train_rows, skip1 = build_split(train_imgs, lm_all, bbox_lookup)
val_rows,   skip2 = build_split(val_imgs,   lm_all, bbox_lookup)
test_rows,  skip3 = build_split(test_imgs,  lm_all, bbox_lookup)
print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
total_skipped = skip1 + skip2 + skip3
if total_skipped:
    print(f"  WARNING: skipped {len(total_skipped)} images")

print(f"Saving to {OUT_PATH}...")
if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)

DatasetDict({
    "train": Dataset.from_list(train_rows),
    "val":   Dataset.from_list(val_rows),
    "test":  Dataset.from_list(test_rows),
}).save_to_disk(OUT_PATH)

print("Done!")
