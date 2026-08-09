"""
benchmark_standalone.py
Benchmark PhoBERT và LayoutLMv3 trực tiếp trên OFFICIAL_LAYOUTLM_DATASET/test
Không cần API server hay PaddleOCR.

Usage:
  python3 benchmark_standalone.py \
    --data_dir /workspace/DoAn/data/OFFICIAL_LAYOUTLM_DATASET \
    --phobert_dir /workspace/phobert_avir_official_best \
    --layoutlm_dir /workspace/layoutlmv3_avir_official_best \
    --out /workspace/benchmark_standalone.csv
"""
import os, sys, json, argparse, csv
import re
from pathlib import Path

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    from difflib import SequenceMatcher
    def levenshtein_distance(a, b):
        sm = SequenceMatcher(None, a, b)
        return int((1 - sm.ratio()) * max(len(a), len(b)))

HEADER_FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
ITEM_FIELDS   = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]
LABEL_MAP     = {"ITEM_QUANTITY": "ITEM_QTY", "ITEM_TOTAL": "ITEM_AMOUNT"}

def normalize(text):
    if not text: return ""
    text = str(text).strip()
    text = re.sub(r'[\s]*(đ|vnd|VND)$', '', text, flags=re.IGNORECASE).strip()
    if re.match(r'^[\d\s,\.]+$', text):
        text = re.sub(r'[,\.\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def ned(pred, gt):
    p, g = normalize(pred), normalize(gt)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    ml = max(len(p), len(g))
    return 1.0 - levenshtein_distance(p, g) / ml

def load_dataset(data_dir):
    test_path = os.path.join(data_dir, "test")
    from datasets import load_from_disk
    return load_from_disk(test_path)

# ──────────────────────────────────────────────
# PhoBERT Inference
# ──────────────────────────────────────────────
def run_phobert(model_dir, dataset):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PhoBERT from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    id2label = model.config.id2label

    MAX_LEN = 256
    results = []
    for sample in dataset:
        words  = sample["tokens"]
        labels_gt = sample["ner_tags"]  # int list

        # Tokenize manually (non-fast tokenizer)
        all_tokens, word_ids = [], []
        for wi, word in enumerate(words):
            sub = tokenizer.tokenize(word) or [tokenizer.unk_token]
            all_tokens.extend(sub)
            word_ids.extend([wi] * len(sub))

        # Truncate
        all_tokens = all_tokens[:MAX_LEN - 2]
        word_ids   = word_ids[:MAX_LEN - 2]

        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(all_tokens) + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        # Pad
        pad_len = MAX_LEN - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
        attn_t      = torch.tensor([attention_mask], dtype=torch.long).to(device)

        # Clamp OOV
        input_ids_t = input_ids_t.clamp(0, model.config.vocab_size - 1)

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Map to word labels (first subword wins)
        word_preds = ["O"] * len(words)
        for tok_i, wi in enumerate(word_ids):
            label_i = preds[tok_i + 1]  # +1 for [CLS]
            if word_preds[wi] == "O":
                raw = id2label.get(label_i, "O")
                word_preds[wi] = LABEL_MAP.get(raw[2:], raw[2:]) if raw != "O" else "O"
                if word_preds[wi] != "O":
                    word_preds[wi] = raw[:2] + word_preds[wi]

        results.append(bio_to_entities(words, word_preds))
    return results


# ──────────────────────────────────────────────
# LayoutLMv3 Inference
# ──────────────────────────────────────────────
def run_layoutlm(model_dir, dataset):
    import torch
    from transformers import AutoProcessor, AutoModelForTokenClassification
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LayoutLMv3 from {model_dir}...")
    try:
        processor = AutoProcessor.from_pretrained(model_dir)
    except Exception:
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    id2label = model.config.id2label

    results = []
    for sample in dataset:
        words   = sample["tokens"]
        bboxes  = sample["bboxes"]
        img_path = sample.get("image_path", "")

        # Normalize bboxes to 0-1000 if needed
        norm_boxes = []
        for b in bboxes:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                norm_boxes.append([int(x) for x in b])
            else:
                norm_boxes.append([0, 0, 0, 0])

        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color=(255, 255, 255))

            enc = processor(image, words, boxes=norm_boxes,
                            return_tensors="pt", truncation=True, max_length=512)
            enc_gpu = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc_gpu).logits
            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            word_ids = enc.word_ids()

            word_preds = ["O"] * len(words)
            for tok_i, wi in enumerate(word_ids):
                if wi is not None and word_preds[wi] == "O":
                    raw = id2label.get(preds[tok_i], "O")
                    if raw != "O":
                        etype = LABEL_MAP.get(raw[2:], raw[2:])
                        word_preds[wi] = raw[:2] + etype

            results.append(bio_to_entities(words, word_preds))
        except Exception as e:
            print(f"  Error on sample: {e}")
            results.append({f: "" for f in HEADER_FIELDS} | {f: [] for f in ITEM_FIELDS})
    return results


def bio_to_entities(words, labels):
    parsed = {f: [] for f in HEADER_FIELDS + ITEM_FIELDS}
    cur_label, cur_words = None, []
    for word, label in zip(words, labels):
        if label != "O":
            bio = label[0]
            etype = label[2:]
            if bio == "B":
                if cur_label: parsed[cur_label].append(" ".join(cur_words))
                cur_label, cur_words = etype, [word]
            elif bio == "I" and cur_label == etype:
                cur_words.append(word)
        else:
            if cur_label: parsed[cur_label].append(" ".join(cur_words))
            cur_label, cur_words = None, []
    if cur_label: parsed[cur_label].append(" ".join(cur_words))

    result = {}
    for f in HEADER_FIELDS:
        result[f] = " ".join(parsed[f]).strip()
    # Build ITEMS list
    items = []
    n = max((len(parsed[f]) for f in ITEM_FIELDS), default=0)
    for i in range(n):
        items.append({f: (parsed[f][i] if i < len(parsed[f]) else "") for f in ITEM_FIELDS})
    result["ITEMS"] = items
    return result


# ──────────────────────────────────────────────
# GT parsing from NER tags
# ──────────────────────────────────────────────
def gt_from_sample(sample, id2label_gt):
    words = sample["tokens"]
    tags  = sample["ner_tags"]
    labels = [id2label_gt.get(t, "O") if isinstance(t, int) else t for t in tags]
    return bio_to_entities(words, labels)


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────
def score_header(pred, gt):
    scores = {}
    for f in HEADER_FIELDS:
        scores[f] = ned(pred.get(f, ""), gt.get(f, ""))
    return scores

def score_items(pred_items, gt_items):
    if not gt_items: return {f: 1.0 for f in ITEM_FIELDS}
    if not pred_items: return {f: 0.0 for f in ITEM_FIELDS}
    matched = set()
    field_scores = {f: [] for f in ITEM_FIELDS}
    for pi in pred_items:
        best_s, best_j = -1, -1
        for j, gi in enumerate(gt_items):
            if j in matched: continue
            s = ned(pi.get("ITEM_NAME",""), gi.get("ITEM_NAME",""))
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            matched.add(best_j)
            gi = gt_items[best_j]
            for f in ITEM_FIELDS:
                field_scores[f].append(ned(pi.get(f,""), gi.get(f,"")))
    # Unmatched GT → 0
    for j, gi in enumerate(gt_items):
        if j not in matched:
            for f in ITEM_FIELDS:
                field_scores[f].append(0.0)
    return {f: (sum(v)/len(v) if v else 0.0) for f, v in field_scores.items()}


def evaluate(predictions, gt_list, id2label_gt):
    all_fields = HEADER_FIELDS + ITEM_FIELDS
    totals = {f: 0.0 for f in all_fields}
    n = len(predictions)

    for pred, sample in zip(predictions, gt_list):
        gt = gt_from_sample(sample, id2label_gt)
        hs = score_header(pred, gt)
        is_ = score_items(pred.get("ITEMS", []), gt.get("ITEMS", []))
        for f in HEADER_FIELDS: totals[f] += hs[f]
        for f in ITEM_FIELDS:   totals[f] += is_[f]

    avg = {f: totals[f] / n for f in all_fields}
    overall = sum(avg.values()) / len(all_fields)
    return avg, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="/workspace/DoAn/data/OFFICIAL_LAYOUTLM_DATASET")
    parser.add_argument("--phobert_dir", default="/workspace/phobert_avir_official_best")
    parser.add_argument("--layoutlm_dir", default="/workspace/layoutlmv3_avir_official_best")
    parser.add_argument("--out",         default="/workspace/benchmark_standalone.csv")
    parser.add_argument("--models",      default="phobert,layoutlm", help="comma-separated: phobert,layoutlm")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.data_dir)
    print(f"Test samples: {len(dataset)}")

    # Get id2label from training labels
    from datasets import load_from_disk
    # Infer label names from features
    features = dataset.features
    if hasattr(features["ner_tags"], "feature"):
        label_names = features["ner_tags"].feature.names
    else:
        label_names = [str(i) for i in range(20)]
    id2label_gt = {i: n for i, n in enumerate(label_names)}
    print(f"Labels: {label_names}")

    models_to_run = [m.strip() for m in args.models.split(",")]
    all_results = {}

    if "phobert" in models_to_run and os.path.exists(args.phobert_dir):
        print("\n=== Running PhoBERT ===")
        preds = run_phobert(args.phobert_dir, dataset)
        avg, overall = evaluate(preds, dataset, id2label_gt)
        all_results["phobert"] = {"avg": avg, "overall": overall}
        print(f"PhoBERT Overall NED: {overall:.4f}")
        for f, s in avg.items(): print(f"  {f}: {s:.4f}")
    else:
        print(f"Skipping PhoBERT (dir not found: {args.phobert_dir})")

    if "layoutlm" in models_to_run and os.path.exists(args.layoutlm_dir):
        print("\n=== Running LayoutLMv3 ===")
        preds = run_layoutlm(args.layoutlm_dir, dataset)
        avg, overall = evaluate(preds, dataset, id2label_gt)
        all_results["layoutlm"] = {"avg": avg, "overall": overall}
        print(f"LayoutLMv3 Overall NED: {overall:.4f}")
        for f, s in avg.items(): print(f"  {f}: {s:.4f}")
    else:
        print(f"Skipping LayoutLMv3 (dir not found: {args.layoutlm_dir})")

    # Save CSV
    all_fields = HEADER_FIELDS + ITEM_FIELDS
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + all_fields + ["overall"])
        for model_name, res in all_results.items():
            row = [model_name] + [f"{res['avg'][field]:.4f}" for field in all_fields] + [f"{res['overall']:.4f}"]
            writer.writerow(row)

    print(f"\nResults saved to {args.out}")
    print("\n=== SUMMARY ===")
    for m, r in all_results.items():
        print(f"{m:15s}  Overall NED: {r['overall']:.4f}")


if __name__ == "__main__":
    main()
