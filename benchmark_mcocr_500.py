import os
import sys
import json
import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm

# Add backend directory to sys.path to import ModelRegistry
backend_dir = "C:/Users/Admin/OneDrive/DoAn/backend"
sys.path.append(backend_dir)
from inference_engine import ModelRegistry

def fuzzy_match(str1, str2):
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    s1 = str(str1).lower().replace(" ", "").strip()
    s2 = str(str2).lower().replace(" ", "").strip()
    return SequenceMatcher(None, s1, s2).ratio()

def main():
    jsonl_path = "C:/Users/Admin/OneDrive/DoAn/test_mcocr.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    print(f"Loaded {len(samples)} samples from {jsonl_path}")
    
    registry = ModelRegistry()
    configs = ["rule_based", "layoutlm", "layoutlmv3", "phobert"]
    
    metrics = {c: {"TP": 0, "FP": 0, "FN": 0} for c in configs}
    
    for config in configs:
        print(f"\n--- Evaluating {config} ---")
        try:
            # We must pass the correct arguments depending on model requirements
            if config == "rule_based":
                registry.load_model(config, None, None)
            elif config == "layoutlm":
                registry.load_model(config, "C:/Users/Admin/OneDrive/DoAn/trained_models/layoutlm-avir-kie-best", None)
            elif config == "layoutlmv3":
                registry.load_model(config, "C:/Users/Admin/OneDrive/DoAn/trained_models/layoutlmv3-avir-kie-best", None)
            elif config == "phobert":
                registry.load_model(config, "C:/Users/Admin/OneDrive/DoAn/trained_models/phobert-avir-kie-best", None)
        except Exception as e:
            print(f"Failed to load {config}: {e}")
            continue

        model = registry.get_model(config)
        
        for sample in tqdm(samples, desc=f"Inference {config}"):
            img_path = sample["image_path"]
            gt_labels = sample["labels"]
            
            try:
                # We need OCR results for all except Qwen2, but this script focuses on the 4 traditional models.
                # Actually, the unified Model.predict interface usually takes (image_path) or requires OCR results depending on the implementation.
                # In inference_engine.py, model.predict() takes (img_path, ocr_data) if it's layoutlm/layoutlmv3/phobert.
                
                # We will just run OCR on the fly
                from inference_engine import run_paddle_ocr
                ocr_data = run_paddle_ocr(img_path)
                
                if config == "rule_based":
                    pred_data = model.predict(img_path, ocr_data)
                elif config == "phobert":
                    # PhoBERT uses only OCR data
                    pred_data = model.predict(ocr_data)
                else:
                    pred_data = model.predict(img_path, ocr_data)
                    
            except Exception as e:
                print(f"Error inferencing {img_path} with {config}: {e}")
                pred_data = {}
            
            all_fields = set(list(gt_labels.keys()) + list(pred_data.keys()))
            for field in all_fields:
                if field == "OTHER" or field == "SELLER": # Ignore SELLER if it's not the target field, but let's just evaluate all fields
                    pass 
                
                gt_val = str(gt_labels.get(field, ""))
                pred_val = str(pred_data.get(field, ""))
                
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
    
    print("\n--- MCOCR 500 SAMPLES BENCHMARK RESULTS ---")
    print(df.to_markdown(index=False))
    
    out_csv = "C:/Users/Admin/OneDrive/DoAn/mcocr_500_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved report to: {out_csv}")

if __name__ == "__main__":
    main()
