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

INPUT_FILE  = "data/FINAL_RUNPOD_DATASET/train.jsonl"
OUTPUT_DIR  = "data/CLEAN_TRAIN_DATASET"
OUTPUT_FILE = f"{OUTPUT_DIR}/train.jsonl"

STANDARD_PROMPT = "<image>Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ... từ hóa đơn này dưới dạng JSON."

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
    field_renames = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

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

    print("=" * 55)
    print("  DATASET CLEAN REPORT")
    print("=" * 55)
    print(f"  Total samples in:            {total:>6}")
    print(f"  Kept (clean):                {kept:>6}  ({100*kept/total:.1f}%)")
    print(f"  Skipped (parse error):       {skipped_no_json:>6}")
    print(f"  Skipped (no SELLER/items):   {skipped_no_fields:>6}")
    print(f"  Samples with renamed fields: {field_renames:>6}")
    print("=" * 55)
    print(f"  Output: {OUTPUT_FILE}")

    # Show 2 sample outputs
    print("\n--- Sample output (first 2 records) ---")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2: break
            rec = json.loads(line)
            print(f"\n[{i+1}] User:      {rec['messages'][0]['content'][:80]}")
            print(f"     Assistant: {rec['messages'][1]['content'][:150]}...")


if __name__ == "__main__":
    main()
