import os
import sys
import json
import glob
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.inference_engine import ModelRegistry

def fuzzy_match(str1, str2):
    if not str1 and not str2: return 1.0
    if not str1 or not str2: return 0.0
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

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

def run_benchmark():
    print("Khởi tạo Model Registry...")
    registry = ModelRegistry()
    
    test_jsonl = r"c:\Users\Admin\OneDrive\DoAn\FINAL_SPLIT_JSONL_ONLY\test.jsonl"
    image_dir = r"c:\Users\Admin\OneDrive\DoAn\FINAL_RUNPOD_DATASET"
    
    if not os.path.exists(test_jsonl):
        print(f"Không tìm thấy file Test: {test_jsonl}")
        return
        
    print(f"Đọc tập Test từ: {test_jsonl}")
    samples = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            samples.append(json.loads(line))
            
    # Lấy 100 sample ngẫu nhiên hoặc tất cả nếu muốn chạy nhanh
    samples = samples[:100] 
    print(f"Bắt đầu Benchmark trên {len(samples)} ảnh (lấy mẫu ngẫu nhiên 100 ảnh)...")
    
    results = []
    fields_to_eval = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
    
    # Models to test
    models = ["layoutlmv1", "phobert", "qwen2"]
    
    for sample in tqdm(samples, desc="Benchmarking"):
        img_filename = sample.get("image", "")
        # The image path is relative in JSONL (e.g. images/...) or just filename?
        # In FINAL_SPLIT_JSONL_ONLY the images are like "einvoice_vnpt_test_30.png" or "images/einvoice..."
        img_basename = os.path.basename(img_filename)
        img_path = os.path.join(image_dir, img_basename)
        
        if not os.path.exists(img_path):
            continue
            
        # Dùng file FINAL_BBOX_DATASET_V3.json để lấy Ground Truth (vì test.jsonl là dạng Text sinh cho VLM)
        # Thay vào đó, trích xuất GT từ câu trả lời có sẵn trong test.jsonl cho nhanh:
        # User answer: "```json\n{...}\n```"
        gt_text = sample["conversations"][1]["value"]
        try:
            import re
            json_match = re.search(r'\{.*\}', gt_text, re.DOTALL)
            gt_json = json.loads(json_match.group(0))
        except:
            gt_json = {}
            
        row = {"image": img_basename}
        
        for m in models:
            try:
                # predict returns result, words, bboxes
                pred_json, _, _ = registry.predict(m, img_path)
            except Exception as e:
                print(f"Lỗi khi chạy {m} trên {img_basename}: {e}")
                pred_json = {}
                
            for field in fields_to_eval:
                gt_val = gt_json.get(field, "")
                pred_val = pred_json.get(field, "")
                score = fuzzy_match(gt_val, pred_val)
                row[f"{m}_{field}_score"] = score
                
        results.append(row)
        
    df = pd.DataFrame(results)
    
    summary = []
    for m in models:
        cols = [f"{m}_{f}_score" for f in fields_to_eval]
        avg_score = df[cols].mean().mean() * 100
        print(f"\n--- Model: {m.upper()} ---")
        for f in fields_to_eval:
            f_score = df[f"{m}_{f}_score"].mean() * 100
            print(f"  {f}: {f_score:.2f}%")
        print(f"  OVERALL: {avg_score:.2f}%")
        summary.append({"Model": m, "Overall Score": avg_score})
        
    df.to_csv("benchmark_qwen_vs_others.csv", index=False)
    print("\nĐã lưu kết quả chi tiết vào benchmark_qwen_vs_others.csv")
    
if __name__ == "__main__":
    run_benchmark()
