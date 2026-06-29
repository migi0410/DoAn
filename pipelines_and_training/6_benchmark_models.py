# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import subprocess
import pandas as pd
from difflib import SequenceMatcher

# Disable PyTorch warnings in subprocesses
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONUTF8"] = "1"

def fuzzy_match(str1, str2):
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    # Normalize strings
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

def get_ground_truth(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    gt = {}
    for ann in data.get('annotations', []):
        lbl = str(ann['label']).upper()
        # Map sub-labels to generic labels
        if lbl.startswith("ITEM_NAME"): lbl = "ITEM_NAME"
        elif lbl.startswith("ITEM_QTY"): lbl = "ITEM_QTY"
        elif lbl.startswith("ITEM_PRICE"): lbl = "ITEM_PRICE"
        elif lbl.startswith("ITEM_AMOUNT"): lbl = "ITEM_AMOUNT"
        elif lbl not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
            continue
            
        if lbl not in gt:
            gt[lbl] = []
        gt[lbl].append(ann['text'])
        
    # Join into strings for easy comparison
    final_gt = {}
    for k, v in gt.items():
        final_gt[k] = " ".join(v).strip()
    return final_gt

def run_inference(config_name, img_path):
    # base_dir is where the actual project scripts are
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is in pipelines_and_training or scratch, we find DoAn root
    # Wait, locally it's in `scratch`, but on Kaggle it might be in `pipelines_and_training`
    # Let's just use dynamic paths relative to the script location or current working dir
    
    # If script is in pipelines_and_training:
    doan_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if not os.path.exists(os.path.join(doan_dir, "pipelines_and_training")):
        # Fallback if run from elsewhere
        doan_dir = os.getcwd()

    base_dir = os.path.join(doan_dir, "pipelines_and_training")
    models_dir = os.path.join(doan_dir, "trained_models")
    
    out_json = os.path.join(base_dir, "temp_inference.json")
    out_img = os.path.join(base_dir, "temp_inference.png")
    
    if os.path.exists(out_json): os.remove(out_json)
    
    cmd = []
    if config_name == "Rule_PaddleOCR":
        cmd = [sys.executable, os.path.join(base_dir, "baseline_rule_based.py"), "--image", img_path, "--ocr", "paddle", "--output_json", out_json]
    elif config_name == "Rule_CRAFT_VietOCR":
        cmd = [sys.executable, os.path.join(base_dir, "baseline_craft_vietocr.py"), "--image", img_path, "--output_json", out_json, "--output_image", out_img]
    elif config_name == "PhoBERT_PaddleOCR":
        m_dir = os.path.join(models_dir, "phobert-avir-kie-best-5k")
        cmd = [sys.executable, os.path.join(base_dir, "5_inference_phobert_v2.py"), "--image", img_path, "--model-dir", m_dir, "--output_json", out_json, "--output_image", out_img]
    elif config_name == "LayoutLM_PaddleOCR":
        m_dir = os.path.join(models_dir, "layoutlm-avir-kie-best-5k")
        cmd = [sys.executable, os.path.join(base_dir, "4_inference_layoutlm.py"), "--image", img_path, "--model-dir", m_dir, "--output_json", out_json, "--output_image", out_img]
    elif config_name == "LayoutLMv3_PaddleOCR":
        m_dir = os.path.join(models_dir, "layoutlmv3-avir-kie-best-5k")
        cmd = [sys.executable, os.path.join(base_dir, "5_inference_layoutlmv3_v2.py"), "--image", img_path, "--model-dir", m_dir, "--ocr", "paddle", "--output_json", out_json, "--output_image", out_img]
    elif config_name == "LayoutLMv3_CRAFT_VietOCR":
        m_dir = os.path.join(models_dir, "layoutlmv3-avir-kie-best-5k")
        cmd = [sys.executable, os.path.join(base_dir, "5_inference_layoutlmv3_v2.py"), "--image", img_path, "--model-dir", m_dir, "--ocr", "craft_vietocr", "--output_json", out_json, "--output_image", out_img]
        
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = doan_dir
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=doan_dir)
        if result.returncode != 0:
            print(f"Error in {config_name}: {result.stderr.strip().split(chr(10))[-1]}")

        
        # Read the absolute path
        out_json_path = out_json
        if os.path.exists(out_json_path):
            with open(out_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error running {config_name}: {e}")
        
    return {}

def main():
    print("=== AVIR-KIE BENCHMARK E2E ===")
    
    doan_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if not os.path.exists(os.path.join(doan_dir, "pipelines_and_training")):
        doan_dir = os.getcwd()
        
    val_images_dir = os.path.join(doan_dir, "synthetic_dataset_v3", "val", "images")
    val_labels_dir = os.path.join(doan_dir, "synthetic_dataset_v3", "val", "labels")
    
    # Get all validation images
    all_imgs = glob.glob(os.path.join(val_images_dir, "*.png"))
    all_imgs.sort()
    
    LIMIT = 2 # Test 2 images to save time for instant slide update
    test_imgs = all_imgs[:LIMIT]
    
    configs = [
        "Rule_PaddleOCR",
        "Rule_CRAFT_VietOCR",
        "PhoBERT_PaddleOCR",
        "LayoutLM_PaddleOCR",
        "LayoutLMv3_PaddleOCR",
        "LayoutLMv3_CRAFT_VietOCR"
    ]
    
    metrics = {c: {"TP": 0, "FP": 0, "FN": 0} for c in configs}
    
    print(f"Bắt đầu đánh giá trên {len(test_imgs)} ảnh validation...")
    
    for idx, img_path in enumerate(test_imgs):
        filename = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, filename.replace(".png", ".json"))
        
        if not os.path.exists(label_path):
            continue
            
        gt_data = get_ground_truth(label_path)
        print(f"[{idx+1}/{len(test_imgs)}] Processing {filename}...")
        
        for config in configs:
            pred_data = run_inference(config, img_path)
            
            # Compare GT and Pred (Macro-average over fields)
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
                    # Fuzzy match threshold 0.8
                    score = fuzzy_match(gt_val, pred_val)
                    if score >= 0.8:
                        metrics[config]["TP"] += 1
                    else:
                        metrics[config]["FP"] += 1
                        metrics[config]["FN"] += 1

    # Calculate final scores
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
    
    out_csv = os.path.join(doan_dir, "pipelines_and_training", "benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n--- KẾT QUẢ BENCHMARK (Fuzzy Match Tương đối >= 80%) ---")
    print(df.to_string())
    print(f"\nĐã xuất báo cáo ra file: {out_csv}")

if __name__ == "__main__":
    main()
