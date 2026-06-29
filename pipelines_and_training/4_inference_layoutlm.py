# 4_inference_layoutlm.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont

# Setup labels mapping
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

LABEL_COLORS = {
    "SELLER": "red",
    "ADDRESS": "blue",
    "TIMESTAMP": "green",
    "TOTAL_COST": "orange",
    "OTHER": "purple"
}

def normalize_bbox(bbox, width, height):
    return [
        max(0, min(1000, int(1000 * (bbox[0] / width)))),
        max(0, min(1000, int(1000 * (bbox[1] / height)))),
        max(0, min(1000, int(1000 * (bbox[2] / width)))),
        max(0, min(1000, int(1000 * (bbox[3] / height))))
    ]

def unnormalize_bbox(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000))
    ]

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

def get_ocr_data(image_path):
    print("Running PaddleOCR to extract text and bounding boxes...")
    from paddleocr import PaddleOCR
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.image_preprocessing import preprocess_for_ocr
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4")
    
    # Preprocess image
    img_array = preprocess_for_ocr(image_path)
    if img_array is not None:
        result = ocr.ocr(img_array)
    else:
        result = ocr.ocr(image_path)
    
    annotations = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]  # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            text = line[1][0]
            
            xmin = min(p[0] for p in box)
            ymin = min(p[1] for p in box)
            xmax = max(p[0] for p in box)
            ymax = max(p[1] for p in box)
            
            annotations.append({
                "text": text,
                "box": [xmin, ymin, xmax, ymax],
                "label": "OTHER" # Dummy label, model will predict the real one
            })
    return annotations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./layoutlm-avir-kie-best")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--label", type=str, default=None, help="Path to paddleocr/synthetic label json (Optional)")
    parser.add_argument("--output_json", type=str, default="inference_result.json")
    parser.add_argument("--output_image", type=str, default="inference_result.png")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    if not os.path.exists(args.model_dir):
        print(f"LỖI: Không tìm thấy thư mục mô hình '{args.model_dir}'. Vui lòng giải nén mô hình tải về từ Kaggle vào đây.")
        return
        
    tokenizer = LayoutLMTokenizerFast.from_pretrained(args.model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = LayoutLMForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    width, height = img.size

    annotations = []
    if args.label:
        print(f"Loading OCR data from {args.label}")
        with open(args.label, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            annotations = label_data.get('annotations', [])
    else:
        annotations = get_ocr_data(args.image)

    words = []
    bboxes = []
    for ann in annotations:
        text = ann['text']
        box = ann['box']
        w_list, box_list = split_box_into_words(text, box)
        for w, b in zip(w_list, box_list):
            words.append(w)
            bboxes.append(normalize_bbox(b, width, height))

    if not words:
        print("No text found in image.")
        return

    # Tokenize
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    # Align bounding boxes
    bbox_tensors = []
    word_ids = encoding.word_ids(batch_index=0)
    for word_idx in word_ids:
        if word_idx is None:
            bbox_tensors.append([0, 0, 0, 0])
        else:
            bbox_tensors.append(bboxes[word_idx])
            
    encoding["bbox"] = torch.tensor([bbox_tensors])
    del encoding["offset_mapping"]
    
    encoding = {k: v.to(device) if hasattr(v, 'to') else v for k, v in encoding.items()}

    # Inference
    print("Running LayoutLM inference...")
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]
        
    token_boxes = encoding["bbox"].squeeze().tolist()
    
    prev_word_idx = None
    
    # --- MỚI: Bóc tách thông tin ra JSON / CSV ---
    word_predicted_labels = ["O"] * len(words)
    for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
        if word_idx is not None and word_predicted_labels[word_idx] == "O":
            word_predicted_labels[word_idx] = id2label[pred]
            
    extracted_data = {
        "SELLER": [],
        "ADDRESS": [],
        "TIMESTAMP": [],
        "TOTAL_COST": [],
        "ITEM_NAME": [],
        "ITEM_QTY": [],
        "ITEM_PRICE": [],
        "ITEM_AMOUNT": []
    }
    
    current_label = None
    current_text = []
    
    for word, label in zip(words, word_predicted_labels):
        if label.startswith("B-"):
            if current_label and current_text:
                extracted_data[current_label].append(" ".join(current_text))
            base = label[2:]
            if base in extracted_data:
                current_label = base
                current_text = [word]
        elif label.startswith("I-") and current_label == label[2:]:
            current_text.append(word)
        else:
            if current_label and current_text:
                extracted_data[current_label].append(" ".join(current_text))
            current_label = None
            current_text = []
            
    if current_label and current_text:
        extracted_data[current_label].append(" ".join(current_text))
        
    # Nối các mảnh chữ lại thành chuỗi hoàn chỉnh
    final_json = {}
    for k, v in extracted_data.items():
        final_json[k] = " ".join(v).strip() if v else None
        
    print("\n--- EXTRACTION RESULT (JSON) ---")
    print(json.dumps(final_json, ensure_ascii=True, indent=4))
    
    json_path = args.output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
        
    # Lưu ra Excel / CSV
    csv_path = args.output_json.replace(".json", ".csv")
    df = pd.DataFrame([final_json])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả dữ liệu ra file {json_path} và {csv_path}")
    # ---------------------------------------------

    # Draw results
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    prev_word_idx = None
    for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
        if word_idx is None or word_idx == prev_word_idx:
            continue
            
        label = id2label[pred]
        if label == "O" or label == "B-OTHER" or label == "I-OTHER":
            prev_word_idx = word_idx
            continue
            
        base_label = label[2:] if "-" in label else label
        color = LABEL_COLORS.get(base_label, "purple")
        
        # unnormalize
        box_1000 = token_boxes[idx]
        actual_box = unnormalize_bbox(box_1000, width, height)
        
        draw.rectangle(actual_box, outline=color, width=3)
        # draw.text((actual_box[0], actual_box[1]-18), label, fill=color, font=font)
        
        prev_word_idx = word_idx

    out_path = "inference_result.png"
    img.save(out_path)
    print(f"Done! Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
