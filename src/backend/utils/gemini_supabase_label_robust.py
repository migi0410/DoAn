import os
import json
import time
import base64
import io
import requests
from PIL import Image
from supabase import create_client

# ================= CẤU HÌNH =================
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
SUPABASE_URL = "https://gmefyvajylsqsyfpuahk.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
BUCKET_NAME = "raw_images"
FOLDERS_TO_PROCESS = ["cafe_starbucks", "supermarket_bachhoaxanh"]
MAX_IMAGES = 5
# ============================================

PROMPT = """
You are an expert OCR and data extraction system.
Analyze this Vietnamese receipt/invoice and extract the following fields.
For each field, provide the extracted text and its bounding box in [ymin, xmin, ymax, xmax] format, where coordinates are normalized between 0 and 1000.

Return the data in STRICT JSON format like this:
{
    "SELLER": {"text": "Name of the store", "box_2d": [ymin, xmin, ymax, xmax]},
    "ADDRESS": {"text": "Address of the store", "box_2d": [ymin, xmin, ymax, xmax]},
    "TIMESTAMP": {"text": "Date and time of the transaction", "box_2d": [ymin, xmin, ymax, xmax]},
    "TOTAL_COST": {"text": "The final total amount paid", "box_2d": [ymin, xmin, ymax, xmax]}
}
If a field is not found, use null for both text and box_2d.
Respond ONLY with the raw JSON string, do not use markdown code blocks like ```json.
"""

def compress_image(image_bytes):
    # Resize image to max 1024px to save bandwidth and reduce 503 errors
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()

def call_gemini_rest_api(image_data):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode('utf-8')
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(8):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print(f"    [!] Server bận (503), thử lại sau {2**attempt}s...")
                time.sleep(2**attempt)
            elif response.status_code == 429:
                print(f"    [!] Quá giới hạn API (429), đợi 15s...")
                time.sleep(15)
            else:
                print(f"    [!] Lỗi API {response.status_code}: {response.text}")
                time.sleep(5)
        except Exception as e:
            print(f"    [!] Lỗi kết nối: {e}, thử lại sau 5s...")
            time.sleep(5)
            
    raise Exception("Lỗi 503/429 liên tục, không thể xử lý.")

def process_images():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    for folder in FOLDERS_TO_PROCESS:
        output_file = f"gemini_labels_{folder}.json"
        print(f"\n==============================================")
        print(f"Đang lấy danh sách ảnh từ folder: {folder}...")
        files = supabase.storage.from_(BUCKET_NAME).list(folder)
        
        image_files = [f['name'] for f in files if f['name'].endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES]
        
        if not image_files:
            print(f"Không tìm thấy ảnh nào trong {folder}!")
            continue

        print(f"Đã tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")
        
        results = {}
        # Load existing if available
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Đã tải {len(results)} kết quả cũ từ {output_file}")
            except:
                pass
        
        for i, filename in enumerate(image_files):
            if filename in results and "error" not in results[filename]:
                print(f"[{i+1}/{len(image_files)}] Bỏ qua (đã có): {filename}")
                continue
                
            print(f"[{i+1}/{len(image_files)}] Đang xử lý: {filename}")
            remote_path = f"{folder}/{filename}"
            
            try:
                image_bytes = supabase.storage.from_(BUCKET_NAME).download(remote_path)
                compressed_bytes = compress_image(image_bytes)
                
                res_json = call_gemini_rest_api(compressed_bytes)
                text = res_json['candidates'][0]['content']['parts'][0]['text']
                
                raw_text = text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:-3].strip()
                    
                data = json.loads(raw_text)
                results[filename] = data
                
                seller = data.get('SELLER', {})
                total = data.get('TOTAL_COST', {})
                seller_text = seller.get('text') if isinstance(seller, dict) else None
                total_text = total.get('text') if isinstance(total, dict) else None
                print(f"  -> Thành công: SELLER='{seller_text}', TOTAL='{total_text}'")
                
            except Exception as e:
                print(f"  -> Lỗi: {e}")
                results[filename] = {"error": str(e)}
                
            # Lưu liên tục
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
            time.sleep(4.5) # Tuân thủ rate limit 15 RPM
            
        print(f"\nHoàn tất folder {folder}! Đã lưu {len(results)} nhãn vào {output_file}")

if __name__ == "__main__":
    process_images()
