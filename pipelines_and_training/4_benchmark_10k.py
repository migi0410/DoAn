# -*- coding: utf-8 -*-
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import io
import time
from difflib import SequenceMatcher
from PIL import Image
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

base_dir = os.path.abspath("C:/Users/Admin/OneDrive/DoAn")
if base_dir not in sys.path:
    sys.path.append(base_dir)

from utils.preprocessing import ImagePreprocessor, TextPreprocessor
def parse_labels_from_predictions(words, word_predicted_labels):
    extracted_data = {
        "STORE_NAME": [], "ADDRESS": [], "DATE": [], "TOTAL_AMOUNT": [],
        "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
    }
    
    current_label = None
    current_text = []
    
    for word, label in zip(words, word_predicted_labels):
        if label.startswith("B-"):
            if current_label and current_text:
                if current_label in extracted_data:
                    extracted_data[current_label].append(" ".join(current_text))
            base = label[2:]
            current_label = base
            current_text = [word]
        elif label.startswith("I-") and current_label:
            base = label[2:]
            if current_label == base:
                current_text.append(word)
        else:
            if current_label and current_text:
                if current_label in extracted_data:
                    extracted_data[current_label].append(" ".join(current_text))
            current_label = None
            current_text = []
            
    if current_label and current_text:
        if current_label in extracted_data:
            extracted_data[current_label].append(" ".join(current_text))
            
    final_json = {}
    for k, v in extracted_data.items():
        if v: final_json[k] = " ".join(v).strip()
    return final_json

# --- Models ---
class RuleModel:
    def predict(self, words, bboxes, img_path, preprocess_text=False):
        if preprocess_text:
            box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
            sorted_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in sorted_dicts]
            
        from baselines.baseline_rule_based import extract_kie_rules
        raw_res = extract_kie_rules(words)
        res = {}
        if "SELLER" in raw_res: res["STORE_NAME"] = raw_res["SELLER"]
        if "TOTAL_COST" in raw_res: res["TOTAL_AMOUNT"] = raw_res["TOTAL_COST"]
        return res

class PhoBertModel:
    def __init__(self, model_dir):
        import torch
        from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizerFast.from_pretrained("vinai/phobert-base-v2", add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        if not words: return {}
        
        if preprocess_text:
            box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
            sorted_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in sorted_dicts]
            
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
        
        word_ids = encoding.word_ids()
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Clamp out-of-vocabulary tokens (due to tokenizer vs model vocab size mismatch)
        vocab_size = self.model.config.vocab_size
        encoding_gpu["input_ids"][encoding_gpu["input_ids"] >= vocab_size] = self.tokenizer.unk_token_id
        
        with torch.no_grad():
            outputs = self.model(**encoding_gpu)
            
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        if isinstance(predictions, int): predictions = [predictions]
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        return parse_labels_from_predictions(words, word_predicted_labels)

class LayoutLMModel:
    def __init__(self, model_dir):
        import torch
        from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_dir)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        if not words: return {}
        
        if preprocess_text:
            box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
            sorted_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in sorted_dicts]
            bboxes = [item["box"] for item in sorted_dicts]
            
        def normalize_bbox(bbox, w, h):
            return [
                max(0, min(1000, int(1000 * (bbox[0] / w)))), max(0, min(1000, int(1000 * (bbox[1] / h)))),
                max(0, min(1000, int(1000 * (bbox[2] / w)))), max(0, min(1000, int(1000 * (bbox[3] / h))))
            ]
        normalized_boxes = [normalize_bbox(b, width, height) for b in bboxes]

        if not words:
            return []
        
        encoding = self.tokenizer(
            words, is_split_into_words=True, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        bbox_tensors = []
        for word_idx in word_ids:
            if word_idx is None:
                bbox_tensors.append([0, 0, 0, 0])
            else:
                bbox_tensors.append(normalized_boxes[word_idx])
                
        encoding["bbox"] = torch.tensor([bbox_tensors])
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        if isinstance(predictions, int): predictions = [predictions]
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_idx < len(word_predicted_labels) and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        return parse_labels_from_predictions(words, word_predicted_labels)

class LayoutLMv3Model:
    def __init__(self, model_dir):
        import torch
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        except Exception as e:
            print(f"LayoutLMv3Processor error: {e}. Falling back to microsoft/layoutlmv3-base")
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        import cv2
        from PIL import Image
        
        box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
        if preprocess_text:
            box_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in box_dicts]
            bboxes = [item["box"] for item in box_dicts]
            
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        
        normalized_boxes = []
        for box in bboxes:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            normalized_boxes.append([
                max(0, min(1000, int(1000 * (x_min / w)))),
                max(0, min(1000, int(1000 * (y_min / h)))),
                max(0, min(1000, int(1000 * (x_max / w)))),
                max(0, min(1000, int(1000 * (y_max / h))))
            ])
            
        encoding = self.processor(
            img_pil,
            words,
            boxes=normalized_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding_gpu)
            
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids()
        
        if isinstance(predictions, int):
            predictions = [predictions]
            
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_idx < len(word_predicted_labels) and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        return parse_labels_from_predictions(words, word_predicted_labels)



def split_box_into_words(text, box):
    words = text.split()
    if not words: return [], []
    x1, y1, x2, y2 = box
    total_len = max(1, sum(len(w) for w in words))
    word_boxes = []
    current_x = x1
    for w in words:
        w_width = (len(w) / total_len) * (x2 - x1)
        word_boxes.append([current_x, y1, current_x + w_width, y2])
        current_x += w_width + (x2 - x1)*0.02
    return words, word_boxes

def load_synthetic_gt(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    words = []
    bboxes = []
    gt_dict = {}
    
    for ann in data.get('annotations', []):
        label = ann['label']
        text = ann['text']
        box = ann['box']
        
        # Add to GT dict
        if label not in gt_dict:
            gt_dict[label] = []
        gt_dict[label].append(text)
        
        w_list, box_list = split_box_into_words(text, box)
        words.extend(w_list)
        bboxes.extend(box_list)
        
    final_gt = {k: " ".join(v).strip() for k, v in gt_dict.items()}
    return words, bboxes, final_gt

def run_benchmark():
    models_dir = os.path.abspath("C:/Users/Admin/OneDrive/DoAn/trained_models")
    print("Loading Models from", models_dir)
    phobert = PhoBertModel(os.path.join(models_dir, "phobert-avir-kie-best-10k"))
    layoutlm = LayoutLMModel(os.path.join(models_dir, "layoutlm-avir-kie-best-10k"))
    
    layoutlmv3_path = os.path.join(models_dir, "layoutlmv3-avir-kie-best-10k")
    layoutlmv3 = LayoutLMv3Model(layoutlmv3_path) if os.path.exists(layoutlmv3_path) else None
    
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, show_log=False)
    
    doan_dir = os.path.abspath("C:/Users/Admin/OneDrive/DoAn")
    val_meta = os.path.join(doan_dir, "synthetic_dataset_hardcore", "metadata.json")
    with open(val_meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    items = []
    for item in meta:
        if item.get("image_path", "").startswith("val"):
            img_path = os.path.join(doan_dir, "synthetic_dataset_hardcore", item["image_path"])
            label_path = os.path.join(doan_dir, "synthetic_dataset_hardcore", item["label_path"])
            if os.path.exists(img_path) and os.path.exists(label_path):
                items.append((img_path, label_path))
                
    import random
    random.seed(42)
    random.shuffle(items)
    # items = items[:100] # Use all 2025 val images
    
    results = {
        "PhoBERT_GT": {"TP": 0, "FP": 0, "FN": 0},
        "PhoBERT_E2E": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLM_GT": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLM_E2E": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLMv3_GT": {"TP": 0, "FP": 0, "FN": 0},
        "LayoutLMv3_E2E": {"TP": 0, "FP": 0, "FN": 0}
    }
    
    def fuzzy_match(s1, s2):
        if not s1 and not s2: return 1.0
        if not s1 or not s2: return 0.0
        s1 = str(s1).lower().replace(" ", "")
        s2 = str(s2).lower().replace(" ", "")
        return SequenceMatcher(None, s1, s2).ratio()
        
    def evaluate(pred, gt, metrics):
        gt_keys = set(gt.keys())
        pred_keys = set(pred.keys())
        for k in gt_keys.union(pred_keys):
            if k not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                continue
            gt_val = gt.get(k, "")
            pred_val = pred.get(k, "")
            
            if gt_val and pred_val:
                if fuzzy_match(gt_val, pred_val) > 0.8:
                    metrics["TP"] += 1
                else:
                    metrics["FP"] += 1
                    metrics["FN"] += 1
            elif gt_val and not pred_val:
                metrics["FN"] += 1
            elif not gt_val and pred_val:
                metrics["FP"] += 1
                
    print(f"Bắt đầu Benchmark trên {len(items)} ảnh...")
    for idx, (img_path, label_path) in enumerate(items):
        if idx % 10 == 0: print(f"Processing {idx}/{len(items)}...")
        
        gt_words, gt_bboxes, gt_data = load_synthetic_gt(label_path)
        
        # --- Mode 1: GT OCR ---
        pred = phobert.predict(gt_words, gt_bboxes, img_path)
        evaluate(pred, gt_data, results["PhoBERT_GT"])
        
        pred = layoutlm.predict(gt_words, gt_bboxes, img_path)
        evaluate(pred, gt_data, results["LayoutLM_GT"])
        
        if layoutlmv3:
            pred = layoutlmv3.predict(gt_words, gt_bboxes, img_path)
            evaluate(pred, gt_data, results["LayoutLMv3_GT"])
            
        # --- Mode 2: E2E Pipeline (OCR) ---
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
        
    print("\n=== KẾT QUẢ BENCHMARK ===")
    for model_name, m in results.items():
        if m["TP"] == 0 and m["FP"] == 0 and m["FN"] == 0: continue
        precision = m["TP"] / (m["TP"] + m["FP"] + 1e-9)
        recall = m["TP"] / (m["TP"] + m["FN"] + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"{model_name:<15} | F1: {f1:.2%}")

if __name__ == "__main__":
    run_benchmark()
