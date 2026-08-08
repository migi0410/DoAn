import os
import json
import time
import base64
import io
import requests
from PIL import Image
from supabase import create_client

# ================= CẤU HÌNH =================
GEMINI_API_KEY = "AIzaSyCjEtuouy2no_Rnu73qUTyjxnlnweFiYl4"
SUPABASE_URL = "https://gmefyvajylsqsyfpuahk.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
BUCKET_NAME = "raw_images"
FOLDER_TO_PROCESS = "supermarket_bachhoaxanh"
MAX_IMAGES = 100
OUTPUT_FILE = f"gemini_labels_{FOLDER_TO_PROCESS}.json"
# ============================================

PROMPT = """
You are an expert OCR and data extraction system.
Analyze this Vietnamese receipt/invoice and extract the following fields in strict JSON format:
{
    "SELLER": "Name of the store, supermarket or company",
    "ADDRESS": "Address of the store",
    "TIMESTAMP": "Date and time of the transaction (e.g. 15/08/2023 14:30:00)",
    "TOTAL_COST": "The final total amount paid (numeric string, e.g. 150000)"
}
If a field is not found, use null.
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
    
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
    
    for attempt in range(5):
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            print(f"    [!] Server bận (503), thử lại sau {2**attempt}s...")
            time.sleep(2**attempt)
        elif response.status_code == 429:
            print(f"    [!] Quá giới hạn API (429), đợi 10s...")
            time.sleep(10)
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
            
    raise Exception("Lỗi 503 liên tục, không thể xử lý.")

def process_images():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print(f"1. Đang kết nối Supabase và lấy danh sách ảnh từ folder: {FOLDER_TO_PROCESS}...")
    files = supabase.storage.from_(BUCKET_NAME).list(FOLDER_TO_PROCESS)
    
    image_files = [f['name'] for f in files if f['name'].endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES]
    
    if not image_files:
        print("Không tìm thấy ảnh nào!")
        return

    print(f"Đã tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý với Gemini 3.5 Flash (Free Tier)...")
    
    results = {}
    
    for i, filename in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Đang xử lý: {filename}")
        remote_path = f"{FOLDER_TO_PROCESS}/{filename}"
        
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
            
            print(f"  -> Thành công: SELLER='{data.get('SELLER')}', TOTAL='{data.get('TOTAL_COST')}'")
            
        except Exception as e:
            print(f"  -> Lỗi: {e}")
            results[filename] = {"error": str(e)}
            
        time.sleep(4.5)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\nHoàn tất! Đã lưu nhãn của {len(results)} ảnh vào {OUTPUT_FILE}")

if __name__ == "__main__":
    process_images()
