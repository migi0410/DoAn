import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import argparse
import pandas as pd
import torch
from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
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

label2color = {
    "SELLER": "blue",
    "ADDRESS": "green",
    "TIMESTAMP": "orange",
    "TOTAL_COST": "red",
    "ITEM_NAME": "purple",
    "ITEM_QTY": "cyan",
    "ITEM_PRICE": "magenta",
    "ITEM_AMOUNT": "brown",
    "OTHER": "gray"
}

def get_ocr_data(image_path):
    print("Running PaddleOCR to extract text and bounding boxes...")
    from paddleocr import PaddleOCR
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.image_preprocessing import preprocess_for_ocr
    
    ocr = PaddleOCR(use_angle_cls=True, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4", show_log=False)
    
    img_array = preprocess_for_ocr(image_path)
    if img_array is not None:
        result = ocr.ocr(img_array)
    else:
        result = ocr.ocr(image_path)
    
    words = []
    boxes = []
    
    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            
            xmin = min(p[0] for p in box)
            ymin = min(p[1] for p in box)
            xmax = max(p[0] for p in box)
            ymax = max(p[1] for p in box)
            
            # Sub-split words by space
            sub_words = text.split()
            if not sub_words: continue
            
            w_width = (xmax - xmin) / max(1, len(text))
            curr_x = xmin
            for w in sub_words:
                w_len = len(w) * w_width
                words.append(w)
                boxes.append([int(curr_x), int(ymin), int(curr_x + w_len), int(ymax)])
                curr_x += w_len + w_width  # Add space
                
    return words, boxes

def get_craft_vietocr_data(image_path):
    print("Running CRAFT + VietOCR to extract text and bounding boxes...")
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    from craft_text_detector import Craft
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    craft = Craft(output_dir=None, crop_type="poly", cuda=(device == 'cuda'))
    
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = device
    detector = Predictor(config)

    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    prediction_result = craft.detect_text(image_path)
    boxes = prediction_result["boxes"]
    
    words = []
    out_boxes = []
    
    for box in boxes:
        rect = cv2.boundingRect(box.astype(np.int32))
        x, y, w, h = rect
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image_rgb.shape[1] - x, w + pad*2)
        h = min(image_rgb.shape[0] - y, h + pad*2)
        cropped = image_rgb[y:y+h, x:x+w]
        
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            continue
            
        cropped_pil = Image.fromarray(cropped)
        try:
            text = detector.predict(cropped_pil)
        except:
            text = ""
            
        if not text.strip():
            continue
            
        sub_words = text.split()
        if not sub_words: continue
        
        w_width = w / max(1, len(text))
        curr_x = x
        for w_sub in sub_words:
            w_len = len(w_sub) * w_width
            words.append(w_sub)
            out_boxes.append([int(curr_x), int(y), int(curr_x + w_len), int(y + h)])
            curr_x += w_len + w_width
            
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    return words, out_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./phobert-avir-kie-best")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--ocr", type=str, choices=["paddle", "craft_vietocr"], default="paddle")
    parser.add_argument("--output_json", type=str, default="inference_phobert_result.json")
    parser.add_argument("--output_image", type=str, default="inference_phobert_result.png")
    args = parser.parse_args()

    print(f"Loading PhoBERT model from {args.model_dir}...")
    if not os.path.exists(args.model_dir):
        print(f"LỖI: Không tìm thấy thư mục mô hình '{args.model_dir}'.")
        return
        
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    width, height = img.size

    if args.ocr == "craft_vietocr":
        words, bboxes = get_craft_vietocr_data(args.image)
    else:
        words, bboxes = get_ocr_data(args.image)

    if not words:
        print("No text found in image.")
        return

    # Tokenize
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    offset_mapping = encoding.pop('offset_mapping')
    
    word_ids = encoding.word_ids()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    encoding_gpu = {k: v.to(device) for k, v in encoding.items()}

    # Inference
    print("Running PhoBERT inference...")
    with torch.no_grad():
        outputs = model(**encoding_gpu)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]
    
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
        
    final_json = {}
    for k, v in extracted_data.items():
        final_json[k] = " ".join(v).strip() if v else None
        
    print("\n--- EXTRACTION RESULT (JSON) ---")
    print(json.dumps(final_json, ensure_ascii=True, indent=4))
    
    json_path = args.output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
        
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
        color = label2color.get(base_label, "purple")
        
        # In PhoBERT, word_idx maps perfectly to the original word array, so we just use the original bbox
        actual_box = bboxes[word_idx]
        
        draw.rectangle(actual_box, outline=color, width=3)
        # draw.text((actual_box[0], actual_box[1]-18), label, fill=color, font=font)
        
        prev_word_idx = word_idx

    out_path = args.output_image
    img.save(out_path)
    print(f"Done! Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
