"""
parse_labelstudio_mcocr.py
Convert Label Studio JSON exports -> JSONL format for benchmark
Usage: python parse_labelstudio_mcocr.py
"""

import json
import re
import os
from collections import defaultdict

INPUT_FILES = [
    r"C:\Users\Admin\Downloads\test2\train_images_1_label.json",
    r"C:\Users\Admin\Downloads\test2\train_images_2_label.json",
    r"C:\Users\Admin\Downloads\test2\val_images_label.json",
]
OUTPUT_JSONL = r"C:\Users\Admin\DoAn\data\test_mcocr_official.jsonl"

PROMPT = "<image>Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON."

# Alias -> canonical
FIELD_ALIASES = {
    "STORE_NAME":    "SELLER",
    "SHOP_NAME":     "SELLER",
    "MERCHANT":      "SELLER",
    "DATE":          "TIMESTAMP",
    "DATETIME":      "TIMESTAMP",
    "TOTAL_AMOUNT":  "TOTAL_COST",
    "GRAND_TOTAL":   "TOTAL_COST",
    "ITEM_QUANTITY": "ITEM_QTY",
}

HEADER_FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
ITEM_FIELDS   = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]


def get_image_name(image_path: str) -> str:
    """Extract filename from Label Studio image path."""
    # /data/local-files/?d=D:/path/to/mcocr_public_xxx.jpg
    m = re.search(r'([^/\\?]+\.(?:jpg|jpeg|png|JPG|JPEG|PNG))$', image_path)
    return m.group(1) if m else os.path.basename(image_path)


def parse_entry(entry: dict) -> dict | None:
    """Parse one Label Studio entry into {SELLER, ADDRESS, ...}."""
    image_path = entry.get("data", {}).get("image", "")
    img_name   = get_image_name(image_path)

    # Collect predictions (may be in 'predictions' or 'annotations')
    results = []
    for pred in entry.get("predictions", []):
        results.extend(pred.get("result", []))
    for ann in entry.get("annotations", []):
        results.extend(ann.get("result", []))

    if not results:
        return None

    # Group by id: {id -> {label: str, text: str, y: float}}
    id_to_label = {}
    id_to_text  = {}
    id_to_y     = {}

    for r in results:
        rid = r.get("id", "")
        rtype = r.get("type", "")
        val   = r.get("value", {})
        y_pos = val.get("y", 0)

        if rtype == "rectanglelabels":
            labels = val.get("rectanglelabels", [])
            if labels:
                raw = labels[0].upper()
                id_to_label[rid] = FIELD_ALIASES.get(raw, raw)
                id_to_y[rid] = y_pos

        elif rtype == "textarea":
            texts = val.get("text", [])
            if texts:
                id_to_text[rid] = texts[0].strip()
                if rid not in id_to_y:
                    id_to_y[rid] = y_pos

    # Group by field
    field_items = defaultdict(list)  # field -> [(y, text)]
    for rid, label in id_to_label.items():
        text = id_to_text.get(rid, "")
        y    = id_to_y.get(rid, 0)
        field_items[label].append((y, text))

    # Sort each field's entries by y (top-to-bottom)
    for field in field_items:
        field_items[field].sort(key=lambda x: x[0])

    # Build output dict
    out = {}

    # Header fields: join multiple lines (ADDRESS can have multiple lines)
    for field in HEADER_FIELDS:
        entries = field_items.get(field, [])
        if entries:
            texts = [t for _, t in entries]
            out[field] = " ".join(texts) if len(texts) > 1 else texts[0]
        else:
            out[field] = ""

    # Item fields: parallel arrays sorted by y
    for field in ITEM_FIELDS:
        entries = field_items.get(field, [])
        out[field] = [t for _, t in entries]

    # Skip records with no useful data
    has_header = any(out.get(f) for f in HEADER_FIELDS)
    has_items  = any(out.get(f) for f in ITEM_FIELDS)
    if not has_header and not has_items:
        return None

    return {"img_name": img_name, "fields": out}


def build_jsonl_record(img_name: str, fields: dict) -> dict:
    """Build JSONL record in OFFICIAL_DATASET format."""
    response = json.dumps(fields, ensure_ascii=False, indent=2)
    assistant_content = f"```json\n{response}\n```"

    return {
        "messages": [
            {"role": "user",      "content": PROMPT},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [f"images/{img_name}"],
    }


def main():
    all_records = []
    skipped = 0

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"  SKIP (not found): {input_file}")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_records = 0
        for entry in data:
            parsed = parse_entry(entry)
            if parsed is None:
                skipped += 1
                continue
            record = build_jsonl_record(parsed["img_name"], parsed["fields"])
            all_records.append(record)
            file_records += 1

        print(f"  {os.path.basename(input_file)}: {file_records} records")

    print(f"\nTotal: {len(all_records)} records  |  Skipped: {skipped}")

    # Write JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUTPUT_JSONL}")

    # Quick sanity check: print first 2 records
    print("\n--- Sample record ---")
    r = all_records[0]
    print("Image:", r["images"])
    gt = json.loads(
        re.search(r'```json\s*(.*?)\s*```', r["messages"][1]["content"], re.DOTALL).group(1)
    )
    for k, v in gt.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
