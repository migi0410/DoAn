"""
benchmark_official.py
Benchmark tất cả models trên OFFICIAL_DATASET/test.jsonl
Chạy: python benchmark_official.py --api http://localhost:8000 --models all
"""

import os
import sys
import json
import re
import csv
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    from difflib import SequenceMatcher
    def levenshtein_distance(a, b):
        \
        sm = SequenceMatcher(None, a, b)
        return int((1 - sm.ratio()) * max(len(a), len(b)))

HEADER_FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
ITEM_FIELDS   = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]
ALL_MODELS    = ["rule_based", "phobert", "layoutlmv3", "minicpm_v", "qwen2_vl"]

\
FIELD_ALIASES = {
    "STORE_NAME":   "SELLER", "SHOP_NAME": "SELLER", "MERCHANT": "SELLER",
    "DATE":         "TIMESTAMP", "DATETIME": "TIMESTAMP", "TIME": "TIMESTAMP",
    "TOTAL_AMOUNT": "TOTAL_COST", "TOTAL": "TOTAL_COST", "GRAND_TOTAL": "TOTAL_COST",
    "ITEM_QUANTITY":"ITEM_QTY",  "QTY": "ITEM_QTY", "SL": "ITEM_QTY",
}

\
\
\
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    \
    text = re.sub(r'[\s]*(đ|vnd|VND)$', '', text, flags=re.IGNORECASE).strip()
    \
    if re.match(r'^[\d\s,\.]+$', text):
        text = re.sub(r'[,\.\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance: 1 - lev_dist / max_len (higher = better)."""
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

def normalize_pred_keys(pred: dict) -> dict:
    """Rename aliased field names to canonical names."""
    result = {}
    for k, v in pred.items():
        canonical = FIELD_ALIASES.get(k.upper(), k.upper())
        result[canonical] = v
    return result

def flatten_items(pred: dict) -> dict:
    """
    Handle both formats:
    - Parallel arrays: ITEM_NAME:[...], ITEM_QTY:[...]
    - ITEMS list: ITEMS:[{ITEM_NAME:..., ITEM_QTY:...}, ...]
    Always output parallel arrays format.
    """
    pred = dict(pred)
    if "ITEMS" in pred and isinstance(pred["ITEMS"], list):
        items = pred["ITEMS"]
        pred["ITEM_NAME"]   = [x.get("ITEM_NAME", x.get("item_name", ""))   for x in items]
        pred["ITEM_QTY"]    = [x.get("ITEM_QTY",  x.get("item_qty",  x.get("QTY", "")))    for x in items]
        pred["ITEM_PRICE"]  = [x.get("ITEM_PRICE", x.get("item_price", ""))  for x in items]
        pred["ITEM_AMOUNT"] = [x.get("ITEM_AMOUNT", x.get("item_amount", "")) for x in items]
        del pred["ITEMS"]
    return pred

def parse_gt_from_record(record: dict) -> dict:
    content = record["messages"][1]["content"]

    \
    m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        \
        content_unescaped = content.replace('\\n', '\n')
        m2 = re.search(r'`json\s*(.*?)\s*`', content_unescaped, re.DOTALL)
        if m2:
            json_str = m2.group(1)
        else:
            start = content_unescaped.find('{')
            end   = content_unescaped.rfind('}')
            json_str = content_unescaped[start:end+1] if start != -1 else ""

    try:
        data = json.loads(json_str)
    except Exception:
        return {}

    data = normalize_pred_keys(data)

    \
    for field in HEADER_FIELDS:
        if isinstance(data.get(field), list):
            data[field] = data[field][0] if data[field] else ""

    return data

def get_image_path(record: dict, images_dirs: list) -> str:
    """Search for image across multiple base directories."""
    images = record.get("images", [])
    if not images:
        return ""
    img = images[0]
    \
    if os.path.isabs(img) and os.path.exists(img):
        return img
    img_name = os.path.basename(img)
    \
    for base in images_dirs:
        candidate = os.path.join(base, img_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(images_dirs[0], img_name) if images_dirs else img_name

def call_api(api_base: str, img_path: str, model: str, timeout: int = 180) -> dict:
    url = f"{api_base}/api/predict"
    try:
        with open(img_path, "rb") as f:
            ext = img_path.rsplit(".", 1)[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            files = {"file": (os.path.basename(img_path), f, mime)}
            data  = {"baseline": model, "preprocess": "false"}
            resp  = requests.post(url, files=files, data=data, timeout=timeout)
        if resp.status_code == 200:
            raw = resp.json().get("result", {})
            raw = normalize_pred_keys(raw)
            raw = flatten_items(raw)
            return raw
        return {}
    except Exception as e:
        return {}

def match_items(pred: dict, gt: dict) -> tuple:
    """
    Match items by ITEM_NAME (NED >= 0.6).
    For each matched pair, also check QTY, PRICE, AMOUNT content.
    Returns (precision, recall, f1, content_accuracy)
    """
    gt_names   = [normalize_text(x) for x in gt.get("ITEM_NAME", [])]
    pred_names = [normalize_text(x) for x in pred.get("ITEM_NAME", [])]

    \
    if not gt_names and not pred_names:
        return 1.0, 1.0, 1.0, 1.0

    if not gt_names or not pred_names:
        return 0.0, 0.0, 0.0, 0.0

    matched_pairs = []
    matched_gt   = set()
    matched_pred = set()
    for i, pn in enumerate(pred_names):
        best_score, best_j = 0.6, -1
        for j, gn in enumerate(gt_names):
            if j in matched_gt:
                continue
            score = ned(pn, gn)
            if score > best_score:
                best_score, best_j = score, j
        if best_j != -1:
            matched_gt.add(best_j)
            matched_pred.add(i)
            matched_pairs.append((i, best_j))

    tp        = len(matched_pairs)
    precision = tp / len(pred_names) if pred_names else 0.0
    recall    = tp / len(gt_names)   if gt_names   else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    \
    content_scores = []
    for pi, gi in matched_pairs:
        field_scores = []
        for field in ["ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]:
            gt_arr   = gt.get(field, [])
            pred_arr = pred.get(field, [])
            gt_val   = gt_arr[gi]   if gi < len(gt_arr)   else ""
            pred_val = pred_arr[pi] if pi < len(pred_arr) else ""
            if gt_val:\
                field_scores.append(ned(pred_val, gt_val))
        if field_scores:
            content_scores.append(sum(field_scores) / len(field_scores))

    content_acc = sum(content_scores) / len(content_scores) if content_scores else 1.0

    return precision, recall, f1, content_acc

def evaluate_sample(pred: dict, gt: dict) -> dict:
    result = {}

    \
    for field in HEADER_FIELDS:
        gt_val   = gt.get(field, "")
        pred_val = pred.get(field, "")
        result[f"{field}_ned"] = ned(pred_val, gt_val)
        result[f"{field}_em"]  = int(exact_match(pred_val, gt_val))

    gt_has_items   = bool(gt.get("ITEM_NAME"))
    pred_has_items = bool(pred.get("ITEM_NAME"))
    result["gt_has_items"] = int(gt_has_items)

    p, r, f1, content_acc = match_items(pred, gt)
    result["item_precision"]    = p
    result["item_recall"]       = r
    result["item_f1"]           = f1
    result["item_content_acc"]  = content_acc

    \
    item_combined = 0.7 * f1 + 0.3 * content_acc
    result["item_combined"] = item_combined

    \
    avg_ned = sum(result[f"{f}_ned"] for f in HEADER_FIELDS) / len(HEADER_FIELDS)
    result["avg_header_ned"] = avg_ned
    result["overall_score"]  = 0.6 * avg_ned + 0.4 * item_combined

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",    default="http://localhost:8000")
    parser.add_argument("--test",   default="data/OFFICIAL_DATASET/test.jsonl")
    parser.add_argument("--images", default="/workspace/FINAL_RUNPOD_DATASET/images,/workspace/bench_images",
                        help="Comma-separated list of image directories to search")
    parser.add_argument("--models", default="all")
    parser.add_argument("--limit",  type=int, default=0)
    parser.add_argument("--out",    default="benchmark_results")
    parser.add_argument("--timeout",type=int, default=180)
    args = parser.parse_args()
    images_dirs = [d.strip() for d in args.images.split(",") if d.strip()]

    models = ALL_MODELS if args.models == "all" else args.models.split(",")

    \
    dataset_specs = []
    for ds_spec in args.test.split(","):
        parts = ds_spec.strip().split(":")
        path  = parts[0]
        limit = int(parts[1]) if len(parts) > 1 else args.limit
        dataset_specs.append((path, limit))

    all_samples = []
    for path, lim in dataset_specs:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_source"] = os.path.basename(path)
                    samples.append(r)
        if lim > 0:
            samples = samples[:lim]
        all_samples.extend(samples)
        print(f"  {os.path.basename(path)}: {len(samples)} samples")

    print(f"Total: {len(all_samples)} samples across {len(dataset_specs)} dataset(s)")

    all_results   = {}
    all_per_sample = {}

    for model in models:
        print(f"\n{'='*55}")
        print(f"  [{model}]")
        print(f"{'='*55}")

        scores     = []
        per_sample = []
        errors     = 0

        for idx, record in enumerate(tqdm(all_samples, desc=model)):
            gt = parse_gt_from_record(record)
            if not gt:
                errors += 1
                continue

            img_path = get_image_path(record, images_dirs)
            if not img_path or not os.path.exists(img_path):
                errors += 1
                continue

            pred  = call_api(args.api, img_path, model, args.timeout)
            score = evaluate_sample(pred if pred else {}, gt)
            score["img"] = os.path.basename(img_path)
            scores.append(score)
            per_sample.append(score)

        if not scores:
            print(f"  No valid samples.")
            continue

        n   = len(scores)
        agg = {k: sum(s[k] for s in scores if isinstance(s[k], (int, float))) / n
               for k in scores[0] if k != "img"}
        agg["n_samples"] = n
        agg["n_errors"]  = errors
        all_results[model]    = agg
        all_per_sample[model] = per_sample

        \
        print(f"\n  Samples={n}  Errors={errors}")
        for field in HEADER_FIELDS:
            print(f"    {field:12s}  NED={agg[f'{field}_ned']:.3f}  EM={agg[f'{field}_em']:.3f}")
        print(f"    Items       P={agg['item_precision']:.3f}  R={agg['item_recall']:.3f}  "
              f"F1={agg['item_f1']:.3f}  Content={agg['item_content_acc']:.3f}")
        print(f"    [OVERALL]   {agg['overall_score']:.3f}")

    print(f"\n\n{'='*85}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*85}")
    hdr = ["Model", "SELLER", "ADDR", "TIME", "COST", "Avg.NED", "Item.F1", "Content", "OVERALL"]
    fmt = "{:<14}" + "{:>9}" * (len(hdr) - 1)
    print(fmt.format(*hdr))
    print("-" * 85)
    sorted_models = sorted(all_results.items(), key=lambda x: -x[1]["overall_score"])
    for model, agg in sorted_models:
        print(fmt.format(
            model,
            f"{agg['SELLER_ned']:.3f}",
            f"{agg['ADDRESS_ned']:.3f}",
            f"{agg['TIMESTAMP_ned']:.3f}",
            f"{agg['TOTAL_COST_ned']:.3f}",
            f"{agg['avg_header_ned']:.3f}",
            f"{agg['item_f1']:.3f}",
            f"{agg['item_content_acc']:.3f}",
            f"{agg['overall_score']:.3f}",
        ))
    print(f"{'='*85}")
    print("Formula: OVERALL = 0.6×Avg.NED + 0.4×(0.7×Item.F1 + 0.3×Item.Content)")

    \
    os.makedirs(args.out, exist_ok=True)

    \
    with open(f"{args.out}/summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if all_results:
        csv_rows = []
        for model, agg in all_results.items():
            csv_rows.append({
                "Model": model,
                "SELLER_NED":    round(agg["SELLER_ned"], 4),
                "ADDRESS_NED":   round(agg["ADDRESS_ned"], 4),
                "TIMESTAMP_NED": round(agg["TIMESTAMP_ned"], 4),
                "TOTAL_COST_NED":round(agg["TOTAL_COST_ned"], 4),
                "Avg_NED":       round(agg["avg_header_ned"], 4),
                "Item_F1":       round(agg["item_f1"], 4),
                "Item_Content":  round(agg["item_content_acc"], 4),
                "Overall":       round(agg["overall_score"], 4),
                "N_Samples":     agg["n_samples"],
                "N_Errors":      agg["n_errors"],
            })
        with open(f"{args.out}/summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    for model, samples_list in all_per_sample.items():
        if not samples_list:
            continue
        with open(f"{args.out}/{model}_per_sample.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=samples_list[0].keys())
            writer.writeheader()
            writer.writerows(samples_list)

    print(f"\nResults saved to: {args.out}/")
    print(f"  summary.json, summary.csv")
    print(f"  <model>_per_sample.csv  (for each model)")

if __name__ == "__main__":
    main()
