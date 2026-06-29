import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import json
import argparse
import time
import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Mock function for regex KIE (same as baseline_rule_based)
def rule_based_kie(text, bbox, img_width, img_height):
    text_upper = text.upper()
    x_min = min(p[0] for p in bbox)
    y_min = min(p[1] for p in bbox)
    y_pct = (y_min / img_height) * 100
    
    if any(k in text_upper for k in ["NGÀY", "GIỜ", "THỜI GIAN", "DATE", "TIME"]):
        return "TIMESTAMP"
    if any(c.isdigit() for c in text_upper) and ("/" in text_upper or ":" in text_upper):
        return "TIMESTAMP"
    if any(k in text_upper for k in ["TỔNG TIỀN", "TỔNG CỘNG", "THANH TOÁN", "CỘNG TIỀN", "TOTAL COST", "GRAND TOTAL"]):
        return "TOTAL_COST"
    if y_pct < 20:
        if any(k in text_upper for k in ["WINMART", "GS25", "COOPMART", "CO.OPMART", "PHARMACITY", "COFFEE", "CAFE", "SIÊU THỊ", "CỬA HÀNG", "NHÀ THUỐC"]):
            return "SELLER"
    if any(k in text_upper for k in ["ĐỊA CHỈ", "Đ/C", "ĐC:", "ADDRESS", "QUẬN", "PHƯỜNG", "ĐƯỜNG", "THÀNH PHỐ", "TP.", "TỈNH"]):
        return "ADDRESS"
    return "OTHER"

def get_vietocr_config():
    from vietocr.tool.config import Cfg
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0' # fallback to cpu if not available in predict
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_image", default=None)
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print("Loading CRAFT and VietOCR...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from craft_text_detector import Craft
        from vietocr.tool.predictor import Predictor
    except ImportError:
        print("Please install craft-text-detector and vietocr")
        return

    # Initialize CRAFT
    craft = Craft(output_dir=None, crop_type="box", cuda=(device == 'cuda'))
    
    # Initialize VietOCR
    config = get_vietocr_config()
    config['device'] = device
    detector = Predictor(config)

    # Load image for CRAFT
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    img_height, img_width = image_rgb.shape[:2]

    # Run CRAFT
    print("Running CRAFT Detection...")
    prediction_result = craft.detect_text(image_path)
    # prediction_result["boxes"] is a list of poly boxes
    boxes = prediction_result["boxes"]
    
    print(f"CRAFT found {len(boxes)} text boxes.")
    
    results = []
    
    colors = {
        "SELLER": (0, 0, 255),       # Red
        "ADDRESS": (255, 0, 0),      # Blue
        "TIMESTAMP": (0, 255, 0),    # Green
        "TOTAL_COST": (0, 165, 255), # Orange
        "OTHER": (128, 128, 128)     # Gray
    }
    
    # Function to crop poly box from image
    def crop_poly(img, pts):
        rect = cv2.boundingRect(pts.astype(np.int32))
        x, y, w, h = rect
        # Expand slightly to avoid cutting off text
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + pad*2)
        h = min(img.shape[0] - y, h + pad*2)
        cropped = img[y:y+h, x:x+w]
        return cropped

    for box in boxes:
        # box is a 4x2 array
        cropped_cv = crop_poly(image_rgb, box)
        if cropped_cv.shape[0] == 0 or cropped_cv.shape[1] == 0:
            continue
            
        cropped_pil = Image.fromarray(cropped_cv)
            
        # Run VietOCR
        try:
            text = detector.predict(cropped_pil)
        except Exception:
            text = ""
            
        if not text.strip():
            continue
            
        kie_label = rule_based_kie(text, box, img_width, img_height)
        
        results.append({
            "text": text,
            "label": kie_label,
            "box": box.tolist()
        })
        
        # Draw on image
        pts = box.astype(np.int32)
        color = colors.get(kie_label, colors["OTHER"])
        cv2.polylines(image_cv, [pts], isClosed=True, color=color, thickness=2)
        
        # if kie_label != "OTHER":
        #     cv2.putText(image_cv, f"{kie_label}", (pts[0][0], pts[0][1] - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    final_dict = {
        "SELLER": [], "ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": [],
        "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
    }
    for r in results:
        label = r['label']
        text = r['text']
        if label in final_dict:
            final_dict[label].append(text)
            
    final_output = {k: ' '.join(v) if v else None for k, v in final_dict.items()}

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
            
    if args.output_image:
        cv2.imwrite(args.output_image, image_cv)
        
    print(f"Extracted {len(results)} items using CRAFT+VietOCR.")

if __name__ == "__main__":
    main()
