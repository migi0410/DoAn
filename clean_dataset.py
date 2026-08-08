"""
clean_dataset.py
Chuẩn hóa train.jsonl trước khi train lại Qwen2-VL LoRA.
- Chuẩn hóa field names
- Chuẩn hóa response format (luôn dùng ```json block)
- Chuẩn hóa prompt (1 prompt duy nhất)
- Báo cáo số sample bị bỏ / sửa
"""

import json
import os
import re
from pathlib import Path

INPUT_DIR   = "data/FINAL_SPLIT_JSONL_ONLY"
OUTPUT_DIR  = "data/OFFICIAL_DATASET"

SPLITS = ["train", "val", "test"]

STANDARD_PROMPT = "<image>Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON."

# Map alias field names -> standard field names
FIELD_ALIASES = {
    "STORE_NAME":    "SELLER",
    "SHOP_NAME":     "SELLER",
    "MERCHANT":      "SELLER",
    "DATE":          "TIMESTAMP",
    "TIME":          "TIMESTAMP",
    "DATETIME":      "TIMESTAMP",
    "TOTAL_AMOUNT":  "TOTAL_COST",
    "TOTAL":         "TOTAL_COST",
    "AMOUNT":        "TOTAL_COST",
    "GRAND_TOTAL":   "TOTAL_COST",
    "ITEM_QUANTITY": "ITEM_QTY",
    "QTY":           "ITEM_QTY",
    "SL":            "ITEM_QTY",
}

def normalize_json_response(raw_text):
    """Extract JSON from response string (raw or in code block), normalize field names, return json block."""
    # 1. Extract JSON string
    json_str = None
    if "```json" in raw_text:
        m = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
    elif "```" in raw_text:
        m = re.search(r'```\s*(.*?)\s*```', raw_text, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
    else:
        # Raw JSON
        start = raw_text.find("{")
        end   = raw_text.rfind("}")
        if start != -1 and end != -1:
            json_str = raw_text[start:end+1].strip()

    if not json_str:
        return None

    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # 3. Normalize field names (case-insensitive alias lookup)
    new_data = {}
    for k, v in data.items():
        k_upper = k.upper()
        canonical = FIELD_ALIASES.get(k_upper, k_upper)
        new_data[canonical] = v

    # 4. Validate: must have SELLER or at least ITEM_NAME
    has_seller = "SELLER" in new_data and new_data["SELLER"]
    has_items  = "ITEM_NAME" in new_data and new_data["ITEM_NAME"]
    if not has_seller and not has_items:
        return None

    # 5. Wrap in ```json block
    return "```json\n" + json.dumps(new_data, ensure_ascii=False, indent=2) + "\n```"


def normalize_user_prompt(content):
    return STANDARD_PROMPT


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    kept = 0
    skipped_no_json = 0
    skipped_no_fields = 0
def process_split(split_name):
    input_file  = f"{INPUT_DIR}/{split_name}.jsonl"
    output_file = f"{OUTPUT_DIR}/{split_name}.jsonl"

    total = kept = skipped_no_json = skipped_no_fields = field_renames = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_no_json += 1
                continue

            messages = record.get("messages", [])
            if len(messages) < 2:
                skipped_no_json += 1
                continue

            user_msg      = messages[0]
            assistant_msg = messages[1]

            # Normalize user prompt
            user_msg["content"] = normalize_user_prompt(user_msg.get("content", ""))

            # Count field renames before normalization
            try:
                raw_json = assistant_msg.get("content", "")
                if "```json" in raw_json:
                    m = re.search(r'```json\s*(.*?)\s*```', raw_json, re.DOTALL)
                    if m: raw_json = m.group(1)
                parsed = json.loads(raw_json.strip())
                for k in parsed.keys():
                    if k.upper() in FIELD_ALIASES:
                        field_renames += 1
                        break
            except Exception:
                pass

            normalized = normalize_json_response(assistant_msg.get("content", ""))
            if normalized is None:
                skipped_no_fields += 1
                continue

            assistant_msg["content"] = normalized
            record["messages"] = [user_msg, assistant_msg]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"  [{split_name:5s}] {total:>5} in -> {kept:>5} kept "
          f"| skip_parse={skipped_no_json} skip_empty={skipped_no_fields} renamed={field_renames}")
    return kept


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  DATASET CLEAN REPORT")
    print("=" * 60)

    totals = {}
    for split in SPLITS:
        totals[split] = process_split(split)

    print("=" * 60)
    print(f"  Total kept: {sum(totals.values())} samples across {len(SPLITS)} splits")
    print(f"  Output dir: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
