# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
from difflib import SequenceMatcher
import io
import cv2
import importlib.util

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Import models from 4_benchmark_10k.py
benchmark_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4_benchmark_10k.py")
spec = importlib.util.spec_from_file_location("benchmark_10k", benchmark_script_path)
benchmark_10k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_10k)

PhoBertModel = benchmark_10k.PhoBertModel
LayoutLMModel = benchmark_10k.LayoutLMModel
LayoutLMv3Model = benchmark_10k.LayoutLMv3Model
ImagePreprocessor = benchmark_10k.ImagePreprocessor

def fuzzy_match(str1, str2):
    if not str1 and not str2: return 1.0
    if not str1 or not str2: return 0.0
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

def extract_ground_truth(task_data):
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
            text = r['value']['text'][0]
            text_map[rid] = text
    for rid, lbl in label_map.items():
        if rid in text_map:
            if lbl not in gt: gt[lbl] = []
            gt[lbl].append(text_map[rid])
    return {k: " ".join(v).strip() for k, v in gt.items()}

def evaluate(pred, gt, metrics):
    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())
    for k in gt_keys.union(pred_keys):
        if k not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]:
            continue
        gt_val = gt.get(k, "")
        pred_val = pred.get(k, "")
        if gt_val and pred_val:
            if fuzzy_match(gt_val, pred_val) > 0.8: metrics["TP"] += 1
            else:
                metrics["FP"] += 1
                metrics["FN"] += 1
        elif gt_val and not pred_val: metrics["FN"] += 1
        elif not gt_val and pred_val: metrics["FP"] += 1

def main():
    print("=== AVIR-KIE BENCHMARK MCOCR (REAL WORLD) ===")
    
    doan_dir = os.path.abspath("C:/Users/Admin/OneDrive/DoAn")
    models_dir = os.path.join(doan_dir, "trained_models")
    
    print("Loading Models...")
    phobert = PhoBertModel(os.path.join(models_dir, "phobert-avir-kie-best-10k"))
    layoutlm = LayoutLMModel(os.path.join(models_dir, "layoutlm-avir-kie-best-10k"))
    layoutlmv3_path = os.path.join(models_dir, "layoutlmv3-avir-kie-best-10k")
    layoutlmv3 = LayoutLMv3Model(layoutlmv3_path) if os.path.exists(layoutlmv3_path) else None
    
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, show_log=False)
    
    json_path = os.path.join(doan_dir, "raw_data", "labels", "Vuong_Label.json")
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    img_base_dir = os.path.join(doan_dir, "label_studio_tasks", "task_vuong", "images")
    
    results = {
        "PhoBERT_E2E": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLM_E2E": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLMv3_E2E": {"TP": 0, "FP": 0, "FN": 0}
    }
    
    print(f"Bắt đầu đánh giá trên {len(tasks)} ảnh thực tế (MCOCR)...")
    for idx, task in enumerate(tasks):
        img_url = task['data']['image']
        filename = img_url.split("/")[-1]
        img_path = os.path.join(img_base_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"[!] Warning: Image {filename} not found in {img_base_dir}. Skipping.")
            continue
            
        gt_data = extract_ground_truth(task)
        if idx % 10 == 0:
            print(f"[{idx}/{len(tasks)}] Processing {filename}...")
        
        img = cv2.imread(img_path)
        processed = ImagePreprocessor.process_all(img)
        temp_img_path = img_path + "_temp.jpg"
        cv2.imwrite(temp_img_path, processed)
        
        ocr_res = ocr.ocr(temp_img_path, cls=True)
        e2e_words, e2e_bboxes = [], []
        if ocr_res and ocr_res[0]:
            for line in ocr_res[0]:
                box = line[0]
                x_c = [p[0] for p in box]
                y_c = [p[1] for p in box]
                e2e_bboxes.append([min(x_c), min(y_c), max(x_c), max(y_c)])
                e2e_words.append(line[1][0])
                
        pred = phobert.predict(e2e_words, e2e_bboxes, temp_img_path, preprocess_text=True)
        evaluate(pred, gt_data, results["PhoBERT_E2E"])
        
        pred = layoutlm.predict(e2e_words, e2e_bboxes, temp_img_path, preprocess_text=True)
        evaluate(pred, gt_data, results["LayoutLM_E2E"])
        
        if layoutlmv3:
            pred = layoutlmv3.predict(e2e_words, e2e_bboxes, temp_img_path, preprocess_text=True)
            evaluate(pred, gt_data, results["LayoutLMv3_E2E"])
            
        if os.path.exists(temp_img_path): os.remove(temp_img_path)

    print("\n=== KẾT QUẢ MCOCR BENCHMARK ===")
    df_res = []
    for model_name, m in results.items():
        if m["TP"] == 0 and m["FP"] == 0 and m["FN"] == 0: continue
        precision = m["TP"] / (m["TP"] + m["FP"] + 1e-9)
        recall = m["TP"] / (m["TP"] + m["FN"] + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"{model_name:<15} | F1: {f1:.2%}")
        df_res.append({
            "Method": model_name,
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        })
        
    df = pd.DataFrame(df_res).sort_values(by="F1-Score", ascending=False)
    out_csv = os.path.join(doan_dir, "pipelines_and_training", "mcocr_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nĐã xuất báo cáo ra file: {out_csv}")

if __name__ == "__main__":
    main()
