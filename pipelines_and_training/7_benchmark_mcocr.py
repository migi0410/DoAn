# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import pandas as pd
from difflib import SequenceMatcher
import io

# Fix stdout encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Disable PyTorch warnings in subprocesses
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONUTF8"] = "1"

def fuzzy_match(str1, str2):
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

def extract_ground_truth(task_data):
    gt = {}
    annotations = task_data.get('annotations', [])
    if not annotations:
        return {}
    
    # Label Studio structure: annotations[0]['result']
    results = annotations[0].get('result', [])
    
    # We need to map labels to text based on ID because they are separate items in Label Studio JSON
    label_map = {} # id -> label
    text_map = {} # id -> text
    
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
            
    # Combine
    for rid, lbl in label_map.items():
        if rid in text_map:
            if lbl not in gt:
                gt[lbl] = []
            gt[lbl].append(text_map[rid])
            
    final_gt = {}
    for k, v in gt.items():
        final_gt[k] = " ".join(v).strip()
    return final_gt

def run_inference(config_name, img_path):
    doan_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if not os.path.exists(os.path.join(doan_dir, "pipelines_and_training")):
        doan_dir = os.getcwd()

    base_dir = os.path.join(doan_dir, "pipelines_and_training")
    models_dir = os.path.join(doan_dir, "trained_models")
    
    out_json = "temp_mcocr.json"
    
    if os.path.exists(out_json): os.remove(out_json)
    
    cmd = []
    if config_name == "Rule_PaddleOCR":
        cmd = [sys.executable, os.path.join(base_dir, "baseline_rule_based.py"), "--image", img_path, "--ocr", "paddle", "--output_json", out_json]
    elif config_name == "LayoutLM_PaddleOCR":
        m_dir = os.path.join(models_dir, "layoutlm-avir-kie-best")
        cmd = [sys.executable, os.path.join(base_dir, "4_inference_layoutlm.py"), "--image", img_path, "--model-dir", m_dir, "--output_json", out_json]
    elif config_name == "LayoutLMv3_PaddleOCR":
        m_dir = os.path.join(models_dir, "layoutlmv3-avir-kie-best")
        cmd = [sys.executable, os.path.join(base_dir, "5_inference_layoutlmv3_v2.py"), "--image", img_path, "--model-dir", m_dir, "--ocr", "paddle", "--output_json", out_json]
    elif config_name == "LayoutLMv3_CRAFT_VietOCR":
        m_dir = os.path.join(models_dir, "layoutlmv3-avir-kie-best")
        cmd = [sys.executable, os.path.join(base_dir, "5_inference_layoutlmv3_v2.py"), "--image", img_path, "--model-dir", m_dir, "--ocr", "craft_vietocr", "--output_json", out_json]
    else:
        return {}
        
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = doan_dir
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=doan_dir)
        if result.returncode != 0:
            print(f"Error in {config_name}: {result.stderr.strip().split(chr(10))[-1]}")
        
        out_json_path = os.path.join(base_dir, out_json)
        if os.path.exists(out_json_path):
            with open(out_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error running {config_name}: {e}")
        
    return {}

def main():
    print("=== AVIR-KIE BENCHMARK MCOCR (REAL WORLD) ===")
    
    doan_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if not os.path.exists(os.path.join(doan_dir, "pipelines_and_training")):
        doan_dir = os.getcwd()
        
    json_path = os.path.join(doan_dir, "Vuong_Label.json")
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    img_base_dir = os.path.join(doan_dir, "label_studio_tasks", "task_vuong", "images")
    
    # LIMIT = 20
    # tasks = tasks[:LIMIT]
    
    configs = [
        "LayoutLM_PaddleOCR",
        "LayoutLMv3_PaddleOCR",
        "LayoutLMv3_CRAFT_VietOCR"
    ]
    
    metrics = {c: {"TP": 0, "FP": 0, "FN": 0} for c in configs}
    
    print(f"Bắt đầu đánh giá trên {len(tasks)} ảnh thực tế (MCOCR)...")
    
    for idx, task in enumerate(tasks):
        # Extract filename from Label Studio path like '/data/local-files/?d=D:/.../mcocr_xxx.jpg'
        img_url = task['data']['image']
        filename = img_url.split("/")[-1]
        
        img_path = os.path.join(img_base_dir, filename)
        if not os.path.exists(img_path):
            print(f"[!] Warning: Image {filename} not found in {img_base_dir}. Skipping.")
            continue
            
        gt_data = extract_ground_truth(task)
        print(f"[{idx+1}/{len(tasks)}] Processing {filename}...")
        
        for config in configs:
            pred_data = run_inference(config, img_path)
            
            all_fields = set(list(gt_data.keys()) + list(pred_data.keys()))
            for field in all_fields:
                if field == "OTHER": continue
                
                gt_val = gt_data.get(field, "")
                pred_val = pred_data.get(field, "")
                
                if not gt_val and pred_val:
                    metrics[config]["FP"] += 1
                elif gt_val and not pred_val:
                    metrics[config]["FN"] += 1
                elif gt_val and pred_val:
                    score = fuzzy_match(gt_val, pred_val)
                    if score >= 0.8:
                        metrics[config]["TP"] += 1
                    else:
                        metrics[config]["FP"] += 1
                        metrics[config]["FN"] += 1

    # Calculate scores
    results = []
    for config in configs:
        tp = metrics[config]["TP"]
        fp = metrics[config]["FP"]
        fn = metrics[config]["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "Method": config,
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    
    print("\n--- KẾT QUẢ MCOCR BENCHMARK (Fuzzy Match >= 80%) ---")
    print(df.to_string(index=False))
    
    out_csv = os.path.join(doan_dir, "pipelines_and_training", "mcocr_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nĐã xuất báo cáo ra file: {out_csv}")

if __name__ == "__main__":
    main()
