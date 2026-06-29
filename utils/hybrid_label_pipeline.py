import os
import json
import time
import base64
import io
import requests
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from supabase import create_client

# ================= CẤU HÌNH =================
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
SUPABASE_URL = "https://gmefyvajylsqsyfpuahk.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
BUCKET_NAME = "raw_images"
FOLDERS_TO_PROCESS = [
    "cafe_starbucks", 
    "supermarket_bachhoaxanh",
    "cafe_highlands",
    "cafe_phuclong",
    "convenience_7eleven",
    "supermarket_winmart",
    "einvoice_viettel",
    "receipt_c45_bb"
]
MAX_IMAGES = 99999
# ============================================

def compress_image(image_bytes, img=None):
    if img is not None:
        # Convert cv2 img (BGR) to bytes
        success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes()
        
    img_pil = Image.open(io.BytesIO(image_bytes))
    img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if img_pil.mode == 'RGBA':
        img_pil = img_pil.convert('RGB')
    out = io.BytesIO()
    img_pil.save(out, format="JPEG", quality=85)
    return out.getvalue()

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Handle OpenCV angle differences depending on version
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Only deskew if the angle is significant
    if abs(angle) > 1.0 and abs(angle) < 89.0:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return img, angle
    return img, 0.0

def run_paddle_ocr(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Deskew before OCR
    img, angle = deskew_image(img)
    
    h, w = img.shape[:2]
    
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
    result = ocr.ocr(img, cls=True)
    
    ocr_texts = []
    if result and result[0]:
        for idx, line in enumerate(result[0]):
            box = line[0]
            txt = line[1][0]
            # Normalize all 4 points to 0-1000
            norm_box = []
            for p in box:
                norm_x = int(p[0] * 1000 / w)
                norm_y = int(p[1] * 1000 / h)
                norm_box.extend([norm_x, norm_y])
                
            ocr_texts.append({
                'paddle_id': idx,
                'text': txt,
                'box_2d': norm_box # [x1, y1, x2, y2, x3, y3, x4, y4]
            })
    return ocr_texts, img, h, w

def call_ollama_hybrid(img, ocr_texts):
    ocr_context = json.dumps([{'paddle_id': t['paddle_id'], 'text': t['text']} for t in ocr_texts], ensure_ascii=False)
    
    PROMPT = f"""
You are an expert OCR and data extraction system.
Analyze this Vietnamese receipt/invoice. I also provide a list of OCR texts extracted by an external engine, which may contain typos, along with their `paddle_id`.

[OCR TEXTS]:
{ocr_context}

Task:
Read the IMAGE carefully. Extract the following fields: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST.
For each field, provide the exact correct text you read from the IMAGE. Then, look at the [OCR TEXTS] list and find the `paddle_id` that best corresponds to the location of this field.

Return STRICT JSON format:
{{
    "SELLER": {{"text": "Exact text from image", "paddle_id": <int>}},
    "ADDRESS": {{"text": "Exact text from image", "paddle_id": <int>}},
    "TIMESTAMP": {{"text": "Exact text from image", "paddle_id": <int>}},
    "TOTAL_COST": {{"text": "Exact text from image", "paddle_id": <int>}}
}}
If a field is not found, use null.
Respond ONLY with raw JSON, no markdown blocks.
"""
    
    # Convert image to Base64
    img_b64 = base64.b64encode(compress_image(None, img)).decode('utf-8')
    
    payload = {
        "model": "minicpm-v",
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1
        }
    }
    
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
    if response.status_code == 200:
        res_data = response.json()
        return res_data.get('response', '{}')
    else:
        raise Exception(f"Ollama API Error {response.status_code}: {response.text}")

def draw_bboxes(img, final_data, h, w, out_name):
    colors = {
        "SELLER": (0, 0, 255),    # Red
        "ADDRESS": (0, 255, 0),   # Green
        "TIMESTAMP": (255, 0, 0), # Blue
        "TOTAL_COST": (0, 255, 255) # Yellow
    }
    
    for key, field_data in final_data.items():
        if not field_data or "box_2d" not in field_data: continue
        box = field_data["box_2d"]
        if not box: continue
        
        if len(box) == 4:
            # Fallback for old format
            ymin, xmin, ymax, xmax = box
            ymin_px, xmin_px = int(ymin * h / 1000), int(xmin * w / 1000)
            ymax_px, xmax_px = int(ymax * h / 1000), int(xmax * w / 1000)
            cv2.rectangle(img, (xmin_px, ymin_px), (xmax_px, ymax_px), color, 2)
            cv2.putText(img, key, (xmin_px, ymin_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif len(box) == 8:
            # 4-point polygon format
            pts = []
            for i in range(0, 8, 2):
                pts.append([int(box[i] * w / 1000), int(box[i+1] * h / 1000)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(img, key, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.imwrite(out_name, img)

def process_images():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    for folder in FOLDERS_TO_PROCESS:
        output_file = f"hybrid_labels_{folder}.json"
        results = {}
        
        # Load existing results to skip processed images
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} existing results from {output_file}")
            except:
                pass
                
        files = supabase.storage.from_(BUCKET_NAME).list(folder)
        image_files = [f['name'] for f in files if f['name'].endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES]
        
        for i, filename in enumerate(image_files):
            if filename in results and "error" not in results[filename]:
                print(f"[{i+1}/{len(image_files)}] Skipping (already processed): {filename}")
                continue
                
            print(f"[{i+1}/{len(image_files)}] Processing: {filename}")
            remote_path = f"{folder}/{filename}"
            
            try:
                image_bytes = supabase.storage.from_(BUCKET_NAME).download(remote_path)
                
                # 1. PaddleOCR (which now includes deskewing)
                ocr_texts, img, h, w = run_paddle_ocr(image_bytes)
                
                # 2. Ollama Hybrid
                res_text = call_ollama_hybrid(img, ocr_texts)
                raw_text = res_text.strip()
                if raw_text.startswith("```json"): raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"): raw_text = raw_text[3:-3].strip()
                
                try:
                    gemini_data = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    print(f"  -> Error: Invalid JSON from Ollama. Text: {raw_text}")
                    continue
                
                # 3. Mapping
                final_data = {}
                for field, data in gemini_data.items():
                    if data and "paddle_id" in data:
                        pid = data["paddle_id"]
                        # Find matching box
                        box = None
                        for t in ocr_texts:
                            if t["paddle_id"] == pid:
                                box = t["box_2d"]
                                break
                        final_data[field] = {
                            "text": data.get("text"),
                            "box_2d": box
                        }
                    else:
                        final_data[field] = None
                        
                results[filename] = final_data
                
                print(f"  -> Success: SELLER='{final_data.get('SELLER', {}).get('text')}'")
                
            except Exception as e:
                print(f"  -> Error: {e}")
                results[filename] = {"error": str(e)}
                
            # Save progressively
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
            time.sleep(4)
            
        print(f"Done with {folder}! Saved {output_file}")

if __name__ == "__main__":
    process_images()
