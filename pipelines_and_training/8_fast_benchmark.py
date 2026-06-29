# -*- coding: utf-8 -*-
import os
import sys
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import glob
import pandas as pd
import numpy as np
import torch
from difflib import SequenceMatcher
from tqdm import tqdm
from PIL import Image

# Thêm utils path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.image_preprocessing import preprocess_for_ocr

# Suppress logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ================= LABELS =================
LABELS = [
    "O",
    "B-SELLER", "I-SELLER",
    "B-ADDRESS", "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP",
    "B-TOTAL_COST", "I-TOTAL_COST",
    "B-ITEM_NAME", "I-ITEM_NAME",
    "B-ITEM_QTY", "I-ITEM_QTY",
    "B-ITEM_PRICE", "I-ITEM_PRICE",
    "B-ITEM_AMOUNT", "I-ITEM_AMOUNT",
    "B-OTHER", "I-OTHER"
]
id2label = {idx: lbl for idx, lbl in enumerate(LABELS)}
label2id = {lbl: idx for idx, lbl in enumerate(LABELS)}

# ================= HELPERS =================
def fuzzy_match(str1, str2):
    if not str1 and not str2: return 1.0
    if not str1 or not str2: return 0.0
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

def normalize_bbox(bbox, width, height):
    return [
        max(0, min(1000, int(1000 * (bbox[0] / width)))),
        max(0, min(1000, int(1000 * (bbox[1] / height)))),
        max(0, min(1000, int(1000 * (bbox[2] / width)))),
        max(0, min(1000, int(1000 * (bbox[3] / height))))
    ]

# ================= GROUND TRUTH =================
def get_synthetic_gt(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = {}
    for ann in data.get('annotations', []):
        lbl = str(ann['label']).upper()
        if lbl.startswith("ITEM_NAME"): lbl = "ITEM_NAME"
        elif lbl.startswith("ITEM_QTY"): lbl = "ITEM_QTY"
        elif lbl.startswith("ITEM_PRICE"): lbl = "ITEM_PRICE"
        elif lbl.startswith("ITEM_AMOUNT"): lbl = "ITEM_AMOUNT"
        elif lbl not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]: continue
        if lbl not in gt: gt[lbl] = []
        gt[lbl].append(ann['text'])
    
    final_gt = {}
    for k, v in gt.items(): final_gt[k] = " ".join(v).strip()
    return final_gt

def extract_label_studio_gt(task_data):
    gt = {}
    annotations = task_data.get('annotations', [])
    if not annotations: return {}
    results = annotations[0].get('result', [])
    label_map = {}
    text_map = {}
    for r in results:
        rid = r.get('id')
        if not rid: continue
        if r['type'] == 'rectanglelabels':
            lbl = r['value']['rectanglelabels'][0]
            if lbl.startswith("ITEM_NAME"): lbl = "ITEM_NAME"
            elif lbl.startswith("ITEM_QTY"): lbl = "ITEM_QTY"
            elif lbl.startswith("ITEM_PRICE"): lbl = "ITEM_PRICE"
            elif lbl.startswith("ITEM_AMOUNT"): lbl = "ITEM_AMOUNT"
            label_map[rid] = lbl
        elif r['type'] == 'textarea':
            text_map[rid] = r['value']['text'][0]
    
    for rid, lbl in label_map.items():
        if rid in text_map:
            if lbl not in gt: gt[lbl] = []
            gt[lbl].append(text_map[rid])
            
    final_gt = {}
    for k, v in gt.items(): final_gt[k] = " ".join(v).strip()
    return final_gt

# ================= INFERENCE ENGINE =================
def extract_fields(predictions, words):
    results = {}
    current_label = None
    current_words = []
    
    for word, pred in zip(words, predictions):
        if pred == "O" or pred == "OTHER" or pred.endswith("OTHER"):
            if current_label:
                if current_label not in results: results[current_label] = []
                results[current_label].append(" ".join(current_words))
            current_label = None
            current_words = []
            continue
            
        prefix = pred[:2]
        label = pred[2:]
        if prefix == "B-":
            if current_label:
                if current_label not in results: results[current_label] = []
                results[current_label].append(" ".join(current_words))
            current_label = label
            current_words = [word]
        elif prefix == "I-":
            if current_label == label:
                current_words.append(word)
            else:
                if current_label:
                    if current_label not in results: results[current_label] = []
                    results[current_label].append(" ".join(current_words))
                current_label = label
                current_words = [word]
                
    if current_label:
        if current_label not in results: results[current_label] = []
        results[current_label].append(" ".join(current_words))
        
    final_results = {}
    for k, v in results.items():
        final_results[k] = " ".join(v).strip()
    return final_results

def split_box_into_words_layoutlm(text, box):
    words = text.split()
    if not words: return [], []
    w_width = (box[2] - box[0]) / len(words)
    w_boxes = []
    for i in range(len(words)):
        w_boxes.append([
            int(box[0] + i * w_width), box[1],
            int(box[0] + (i + 1) * w_width), box[3]
        ])
    return words, w_boxes

def predict_layoutlm(model, tokenizer, device, img_path, ocr_results):
    from PIL import Image
    try:
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
    except:
        return {}
        
    words = []
    boxes = []
    for res in ocr_results:
        box = res[0]
        text = res[1][0]
        x_min = min([p[0] for p in box])
        y_min = min([p[1] for p in box])
        x_max = max([p[0] for p in box])
        y_max = max([p[1] for p in box])
        b_words, b_boxes = split_box_into_words_layoutlm(text, [x_min, y_min, x_max, y_max])
        for w, b in zip(b_words, b_boxes):
            words.append(w)
            boxes.append(normalize_bbox(b, width, height))
            
    if not words: return {}
    
    input_ids = [tokenizer.cls_token_id]
    token_boxes = [[0, 0, 0, 0]]
    word_ids = []
    
    for word_idx, (word, box) in enumerate(zip(words, boxes)):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if not tokens: continue
        input_ids.extend(tokens)
        token_boxes.extend([box] * len(tokens))
        word_ids.extend([word_idx] + [None] * (len(tokens) - 1))
        
    input_ids.append(tokenizer.sep_token_id)
    token_boxes.append([1000, 1000, 1000, 1000])
    word_ids.append(None)
    word_ids = [None] + word_ids
    
    max_len = 512
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len-1] + [tokenizer.sep_token_id]
        token_boxes = token_boxes[:max_len-1] + [[1000, 1000, 1000, 1000]]
        word_ids = word_ids[:max_len]
        
    attention_mask = [1] * len(input_ids)
    
    t_input_ids = torch.tensor([input_ids]).to(device)
    t_attention_mask = torch.tensor([attention_mask]).to(device)
    t_bbox = torch.tensor([token_boxes]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=t_input_ids, attention_mask=t_attention_mask, bbox=t_bbox)
        
    preds = torch.argmax(outputs.logits.squeeze(), dim=1).tolist()
    
    word_preds = []
    for w_idx in range(len(words)):
        try:
            token_idx = word_ids.index(w_idx)
            word_preds.append(id2label[preds[token_idx]])
        except:
            word_preds.append("O")
            
    return extract_fields(word_preds, words)

def predict_layoutlmv3(model, processor, device, img_path, ocr_results):
    from PIL import Image
    try:
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
    except:
        return {}
        
    words = []
    boxes = []
    for res in ocr_results:
        box = res[0]
        text = res[1][0]
        x_min = min([p[0] for p in box])
        y_min = min([p[1] for p in box])
        x_max = max([p[0] for p in box])
        y_max = max([p[1] for p in box])
        b_words, b_boxes = split_box_into_words_layoutlm(text, [x_min, y_min, x_max, y_max])
        for w, b in zip(b_words, b_boxes):
            words.append(w)
            boxes.append(normalize_bbox(b, width, height))
            
    if not words: return {}
    
    if len(words) > 400:
        words = words[:400]
        boxes = boxes[:400]
        
    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, max_length=512)
    word_ids = encoding.word_ids()
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        
    preds = torch.argmax(outputs.logits.squeeze(), dim=1).tolist()
    
    word_preds = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None: continue
        if word_idx != previous_word_idx:
            word_preds.append(id2label[preds[idx]])
        previous_word_idx = word_idx
        
    if len(word_preds) > len(words): word_preds = word_preds[:len(words)]
    while len(word_preds) < len(words): word_preds.append("O")
        
    return extract_fields(word_preds, words)

def predict_phobert(model, tokenizer, device, img_path, ocr_results):
    words = []
    for res in ocr_results:
        text = res[1][0]
        words.extend(text.split())
        
    if not words: return {}
    
    input_ids = [tokenizer.cls_token_id]
    word_ids = []
    for word_idx, word in enumerate(words):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if not tokens: continue
        input_ids.extend(tokens)
        word_ids.extend([word_idx] + [None] * (len(tokens) - 1))
        
    input_ids.append(tokenizer.sep_token_id)
    word_ids = [None] + word_ids + [None]
    
    max_len = 256
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len-1] + [tokenizer.sep_token_id]
        word_ids = word_ids[:max_len]
        
    attention_mask = [1] * len(input_ids)
    
    t_input_ids = torch.tensor([input_ids]).to(device)
    t_attention_mask = torch.tensor([attention_mask]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=t_input_ids, attention_mask=t_attention_mask)
        
    preds = torch.argmax(outputs.logits.squeeze(), dim=1).tolist()
    
    word_preds = []
    for w_idx in range(len(words)):
        try:
            token_idx = word_ids.index(w_idx)
            word_preds.append(id2label[preds[token_idx]])
        except:
            word_preds.append("O")
            
    return extract_fields(word_preds, words)

# ================= MAIN =================
def main():
    print("=== AVIR-KIE FAST MULTI-DATASET BENCHMARK ===")
    doan_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    
    datasets = {}
    
    # 1. Val Dataset
    val_images_dir = os.path.join(doan_dir, "synthetic_dataset_v3", "val", "images")
    val_labels_dir = os.path.join(doan_dir, "synthetic_dataset_v3", "val", "labels")
    if os.path.exists(val_images_dir):
        imgs = glob.glob(os.path.join(val_images_dir, "*.png"))[:100] # Limiting for speed, remove [:100] to test all
        data = []
        for img in imgs:
            lbl = os.path.join(val_labels_dir, os.path.basename(img).replace(".png", ".json"))
            if os.path.exists(lbl):
                data.append({"image": img, "gt": get_synthetic_gt(lbl)})
        datasets[f"1. Synthetic_Val ({len(data)} ảnh)"] = data
        
    # 2. MCOCR Vuong
    vuong_json = os.path.join(doan_dir, "Vuong_Label.json")
    if os.path.exists(vuong_json):
        with open(vuong_json, 'r', encoding='utf-8') as f: v_data = json.load(f)
        data = []
        for item in v_data:
            img_name = os.path.basename(item['data']['image'].replace('\\', '/'))
            img_path = os.path.join(doan_dir, "label_studio_tasks", "task_vuong", "images", img_name)
            if os.path.exists(img_path): data.append({"image": img_path, "gt": extract_label_studio_gt(item)})
        datasets[f"2. MCOCR_Vuong ({len(data)} ảnh)"] = data
        
    # 3. MCOCR Cam
    cam_json = os.path.join(doan_dir, "Cam_Label.json")
    if os.path.exists(cam_json):
        with open(cam_json, 'r', encoding='utf-8') as f: c_data = json.load(f)
        data = []
        for item in c_data:
            img_name = os.path.basename(item['data']['ocr'][0]['original_image_url'].replace('\\', '/')) if 'ocr' in item['data'] else os.path.basename(item['data']['image'].replace('\\', '/'))
            img_path = os.path.join(doan_dir, "label_studio_tasks", "task_cam", "images", img_name)
            if os.path.exists(img_path): data.append({"image": img_path, "gt": extract_label_studio_gt(item)})
        datasets[f"3. MCOCR_Cam ({len(data)} ảnh)"] = data
        
    total_imgs = sum([len(v) for v in datasets.values()])
    print(f"Tổng số ảnh cần Benchmark: {total_imgs}")
    
    # 4. OCR Caching
    print("\n--- BƯỚC 1: CACHE PADDLE OCR (Khởi tạo 1 lần) ---")
    
    ocr_cache_file = os.path.join(doan_dir, "pipelines_and_training", "ocr_cache.json")
    ocr_cache = {}
    if os.path.exists(ocr_cache_file):
        with open(ocr_cache_file, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)
            print(f"Loaded {len(ocr_cache)} cached OCR results from disk.")
            
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4", show_log=False)
    
    for ds_name, items in datasets.items():
        print(f"Running OCR cho {ds_name}...")
        save_counter = 0
        for item in tqdm(items, desc=f"OCR {ds_name}"):
            img_path = item["image"]
            if img_path in ocr_cache: continue
            
            try:
                res = ocr.ocr(img_path)
                ocr_cache[img_path] = res[0] if res and len(res) > 0 and res[0] else []
            except Exception as e:
                print(f"Error OCR {img_path}: {e}")
                ocr_cache[img_path] = []
                
            save_counter += 1
            if save_counter >= 20:
                with open(ocr_cache_file, "w", encoding="utf-8") as f:
                    json.dump(ocr_cache, f, ensure_ascii=False)
                save_counter = 0
                
        # Save at end of dataset
        with open(ocr_cache_file, "w", encoding="utf-8") as f:
            json.dump(ocr_cache, f, ensure_ascii=False)
                
    # 5. Benchmark Models
    print("\n--- BƯỚC 2: BENCHMARK MODELS TRÊN TOÀN BỘ TẬP DỮ LIỆU ---")
    models_dir = os.path.join(doan_dir, "trained_models")
    
    model_configs = [
        ("LayoutLM-5k", os.path.join(models_dir, "layoutlm-avir-kie-best-5k"), "layoutlm"),
        ("LayoutLM-10k", os.path.join(models_dir, "layoutlm-avir-kie-best-10k"), "layoutlm"),
        ("LayoutLMv3-5k", os.path.join(models_dir, "layoutlmv3-avir-kie-best-5k"), "layoutlmv3"),
        ("LayoutLMv3-10k", os.path.join(models_dir, "layoutlmv3-avir-kie-best-10k"), "layoutlmv3"),
        ("PhoBERT-5k", os.path.join(models_dir, "phobert-avir-kie-best-5k"), "phobert")
    ]
    
    results_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from transformers import (
        LayoutLMTokenizerFast, LayoutLMForTokenClassification,
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        AutoTokenizer, AutoModelForTokenClassification
    )
    
    for m_name, m_path, m_type in model_configs:
        if not os.path.exists(m_path):
            print(f"Bỏ qua {m_name} (Không tìm thấy folder)")
            continue
            
        print(f"\n=> Đang Load Model: {m_name}")
        model = None
        tokenizer = None
        processor = None
        
        if m_type == "layoutlm":
            tokenizer = LayoutLMTokenizerFast.from_pretrained(m_path)
            model = LayoutLMForTokenClassification.from_pretrained(m_path).to(device)
        elif m_type == "layoutlmv3":
            try:
                processor = LayoutLMv3Processor.from_pretrained(m_path, apply_ocr=False)
            except Exception:
                processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            model = LayoutLMv3ForTokenClassification.from_pretrained(m_path).to(device)
        elif m_type == "phobert":
            tokenizer = AutoTokenizer.from_pretrained(m_path)
            model = AutoModelForTokenClassification.from_pretrained(m_path).to(device)
            
        model.eval()
        
        for ds_name, items in datasets.items():
            tp, fp, fn = 0, 0, 0
            
            for item in tqdm(items, desc=f"Eval {ds_name} - {m_name}"):
                img_path = item["image"]
                gt_data = item["gt"]
                ocr_results = ocr_cache.get(img_path, [])
                
                if not ocr_results: continue
                
                if m_type == "layoutlm": pred_data = predict_layoutlm(model, tokenizer, device, img_path, ocr_results)
                elif m_type == "layoutlmv3": pred_data = predict_layoutlmv3(model, processor, device, img_path, ocr_results)
                elif m_type == "phobert": pred_data = predict_phobert(model, tokenizer, device, img_path, ocr_results)
                    
                for field in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                    gt_val = gt_data.get(field, "")
                    pred_val = pred_data.get(field, "")
                    
                    if not gt_val and pred_val: fp += 1
                    elif gt_val and not pred_val: fn += 1
                    elif gt_val and pred_val:
                        score = fuzzy_match(gt_val, pred_val)
                        if score >= 0.8: tp += 1
                        else:
                            fp += 1
                            fn += 1
                            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results_list.append({
                "Model": m_name,
                "Dataset": ds_name,
                "Precision": round(precision * 100, 2),
                "Recall": round(recall * 100, 2),
                "F1-Score": round(f1 * 100, 2)
            })
            
        del model
        if tokenizer: del tokenizer
        if processor: del processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    # Xuất file CSV
    df = pd.DataFrame(results_list)
    # df = df.sort_values(by=["Dataset", "F1-Score"], ascending=[True, False]).reset_index(drop=True)
    df = df.sort_values(by=["Dataset", "Model"], ascending=[True, True]).reset_index(drop=True)
    
    out_csv = os.path.join(doan_dir, "pipelines_and_training", "fast_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n\n================ KẾT QUẢ BENCHMARK SIÊU TỐC ================")
    print(df.to_string())
    print(f"\nĐã xuất báo cáo CSV ra file: {out_csv}")

if __name__ == "__main__":
    main()
