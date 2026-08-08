"""
benchmark_official.py
Benchmark tất cả models trên OFFICIAL_DATASET/test.jsonl
Chạy: python benchmark_official.py --api http://localhost:8000 --models all
"""

import os
import sys
import json
import re
import argparse
import requests
import unicodedata
from pathlib import Path
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance

# =============================================
# CONFIG
# =============================================
HEADER_FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
ITEM_FIELDS   = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]
ALL_MODELS    = ["rule_based", "phobert", "layoutlmv3", "minicpm_v", "qwen2_vl"]

# =============================================
# TEXT NORMALIZATION
# =============================================
def normalize_text(text: str) -> str:
    """Lowercase, strip accents preserved, normalize whitespace and numbers."""
    if not text:
        return ""
    text = str(text).strip()
    # Remove currency suffix (đ, VND, vnd...)
    text = re.sub(r'[\s]*(đ|vnd|VND)$', '', text, flags=re.IGNORECASE).strip()
    # Normalize thousands separators: 210,600 or 210.600 -> 210600
    # Only if it's clearly a number pattern
    if re.match(r'^[\d,\.]+$', text.replace(' ', '')):
        text = re.sub(r'[,\.]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance: 1 - lev_dist / max_len (higher=better)."""
    p = normalize_text(pred)
    g = normalize_text(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    max_len = max(len(p), len(g))
    return 1.0 - levenshtein_distance(p, g) / max_len

def exact_match(pred: str, gt: str) -> bool:
    return normalize_text(pred) == normalize_text(gt)

# =============================================
# PARSE GROUND TRUTH FROM JSONL
# =============================================
def parse_gt_from_record(record: dict) -> dict:
    """Extract GT fields from messages[1].content (```json block)."""
    content = record["messages"][1]["content"]
    # Strip ```json ... ``` wrapper
    m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        start = content.find('{')
        end   = content.rfind('}')
        json_str = content[start:end+1] if start != -1 else ""
    try:
        return json.loads(json_str)
    except Exception:
        return {}

def get_image_path(record: dict, images_base: str) -> str:
    """Resolve image path from record."""
    images = record.get("images", [])
    if not images:
        return ""
    img = images[0]
    if os.path.isabs(img):
        return img
    # Relative path: resolve from images_base
    img_name = os.path.basename(img)
    return os.path.join(images_base, img_name)

# =============================================
# CALL API
# =============================================
def call_api(api_base: str, img_path: str, model: str) -> dict:
    """Call the FastAPI inference endpoint."""
    url = f"{api_base}/api/predict"
    try:
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
            data  = {"baseline": model, "preprocess": "false"}
            resp  = requests.post(url, files=files, data=data, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("result", {})
        else:
            return {}
    except Exception as e:
        return {}

# =============================================
# ITEM MATCHING
# =============================================
def match_items(pred: dict, gt: dict):
    """
    Match predicted items to GT items by ITEM_NAME (normalized).
    Returns (precision, recall, f1).
    """
    gt_names   = [normalize_text(x) for x in gt.get("ITEM_NAME", [])]
    pred_names = [normalize_text(x) for x in pred.get("ITEM_NAME", [])]

    if not gt_names and not pred_names:
        return 1.0, 1.0, 1.0
    if not gt_names or not pred_names:
        return 0.0, 0.0, 0.0

    # Greedy matching by NED >= 0.6
    matched_gt   = set()
    matched_pred = set()
    for i, pn in enumerate(pred_names):
        best_score = 0.6  # threshold
        best_j     = -1
        for j, gn in enumerate(gt_names):
            if j in matched_gt:
                continue
            score = ned(pn, gn)
            if score > best_score:
                best_score = score
                best_j     = j
        if best_j != -1:
            matched_gt.add(best_j)
            matched_pred.add(i)

    tp = len(matched_pred)
    precision = tp / len(pred_names) if pred_names else 0.0
    recall    = tp / len(gt_names)   if gt_names   else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

# =============================================
# EVALUATE ONE SAMPLE
# =============================================
def evaluate_sample(pred: dict, gt: dict) -> dict:
    """Compute all metrics for one sample."""
    result = {}

    # Header NED + EM per field
    for field in HEADER_FIELDS:
        gt_val   = gt.get(field, "")
        pred_val = pred.get(field, "")
        result[f"{field}_ned"] = ned(pred_val, gt_val)
        result[f"{field}_em"]  = int(exact_match(pred_val, gt_val))

    # Items: handle both parallel-array format and ITEMS-list format
    # Convert ITEMS list to parallel arrays if needed
    if "ITEMS" in pred and isinstance(pred["ITEMS"], list):
        items = pred["ITEMS"]
        pred = dict(pred)
        pred["ITEM_NAME"]   = [x.get("ITEM_NAME", "")   for x in items]
        pred["ITEM_QTY"]    = [x.get("ITEM_QTY", "")    for x in items]
        pred["ITEM_PRICE"]  = [x.get("ITEM_PRICE", "")  for x in items]
        pred["ITEM_AMOUNT"] = [x.get("ITEM_AMOUNT", "") for x in items]

    p, r, f1 = match_items(pred, gt)
    result["item_precision"] = p
    result["item_recall"]    = r
    result["item_f1"]        = f1

    # Overall score: 60% avg header NED + 40% item F1
    avg_ned = sum(result[f"{f}_ned"] for f in HEADER_FIELDS) / len(HEADER_FIELDS)
    result["avg_header_ned"] = avg_ned
    result["overall_score"]  = 0.6 * avg_ned + 0.4 * f1

    return result

# =============================================
# MAIN
# =============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",    default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--test",   default="data/OFFICIAL_DATASET/test.jsonl")
    parser.add_argument("--images", default="/workspace/FINAL_RUNPOD_DATASET/images",
                        help="Absolute path to images directory on RunPod")
    parser.add_argument("--models", default="all",
                        help="Comma-separated list of models, or 'all'")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Limit number of samples (0 = all)")
    parser.add_argument("--out",    default="benchmark_results.json")
    args = parser.parse_args()

    models = ALL_MODELS if args.models == "all" else args.models.split(",")

    # Load test data
    samples = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} test samples")

    all_results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {model}")
        print(f"{'='*50}")

        scores = []
        errors = 0

        for record in tqdm(samples, desc=model):
            gt = parse_gt_from_record(record)
            if not gt:
                errors += 1
                continue

            img_path = get_image_path(record, args.images)
            if not os.path.exists(img_path):
                errors += 1
                continue

            pred = call_api(args.api, img_path, model)
            if not pred:
                errors += 1
                # Count as zero score
                pred = {}

            score = evaluate_sample(pred, gt)
            scores.append(score)

        if not scores:
            print(f"  No valid samples for {model}")
            continue

        # Aggregate
        n = len(scores)
        agg = {}
        for key in scores[0].keys():
            agg[key] = sum(s[key] for s in scores) / n

        agg["n_samples"] = n
        agg["n_errors"]  = errors
        all_results[model] = agg

        # Print summary
        print(f"\n  Results ({n} samples, {errors} errors):")
        for field in HEADER_FIELDS:
            print(f"    {field:12s} NED={agg[f'{field}_ned']:.3f}  EM={agg[f'{field}_em']:.3f}")
        print(f"    Items      P={agg['item_precision']:.3f}  R={agg['item_recall']:.3f}  F1={agg['item_f1']:.3f}")
        print(f"    [OVERALL]  {agg['overall_score']:.3f}  (Header NED={agg['avg_header_ned']:.3f})")

    # ---- FINAL TABLE ----
    print(f"\n\n{'='*80}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    headers = ["Model", "SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST",
               "Avg.NED", "Item.F1", "OVERALL"]
    row_fmt = "{:<14}" + "{:>9}" * (len(headers) - 1)
    print(row_fmt.format(*headers))
    print("-" * 80)

    rows = []
    for model, agg in all_results.items():
        row = {
            "Model":      model,
            "SELLER":     f"{agg['SELLER_ned']:.3f}",
            "ADDRESS":    f"{agg['ADDRESS_ned']:.3f}",
            "TIMESTAMP":  f"{agg['TIMESTAMP_ned']:.3f}",
            "TOTAL_COST": f"{agg['TOTAL_COST_ned']:.3f}",
            "Avg.NED":    f"{agg['avg_header_ned']:.3f}",
            "Item.F1":    f"{agg['item_f1']:.3f}",
            "OVERALL":    f"{agg['overall_score']:.3f}",
        }
        rows.append(row)
        print(row_fmt.format(*[row[h] for h in headers]))

    print(f"{'='*80}")
    print(f"\nFormula: OVERALL = 0.6 × Avg.NED + 0.4 × Item.F1")

    # Save JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()
