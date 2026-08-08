import os
import json
import base64
import io
from PIL import Image
from supabase import create_client
from google import genai
from google.genai import types

# ================= CẤU HÌNH =================
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Admin\Downloads\project-8f1fc845-8eb8-4d08-b90-577457588159.json"
PROJECT_ID = "project-8f1fc845-8eb8-4d08-b90"
LOCATION = "us-central1"

SUPABASE_URL = "https://gmefyvajylsqsyfpuahk.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
BUCKET_NAME = "raw_images"

FOLDERS_TO_PROCESS = ["cafe_starbucks", "supermarket_bachhoaxanh"]
MAX_IMAGES_PER_FOLDER = 100
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
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()

def process_images():
    print("Khởi tạo Vertex AI (google-genai SDK)...")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    for folder in FOLDERS_TO_PROCESS:
        print(f"\n======================================")
        print(f"Bắt đầu xử lý folder: {folder}")
        
        output_file = f"vertexai_labels_{folder}.json"
        
        files = supabase.storage.from_(BUCKET_NAME).list(folder)
        image_files = [f['name'] for f in files if f['name'].endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES_PER_FOLDER]
        
        if not image_files:
            print(f"Không tìm thấy ảnh nào trong {folder}!")
            continue

        print(f"Đã tìm thấy {len(image_files)} ảnh. Đang chạy Vertex AI...")
        
        results = {}
        
        for i, filename in enumerate(image_files):
            print(f"[{i+1}/{len(image_files)}] {filename}", end=" ")
            remote_path = f"{folder}/{filename}"
            
            try:
                # Tải ảnh từ Supabase vào memory
                image_bytes = supabase.storage.from_(BUCKET_NAME).download(remote_path)
                compressed_bytes = compress_image(image_bytes)
                mime_type = "image/jpeg"
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[
                        types.Part.from_bytes(data=compressed_bytes, mime_type=mime_type),
                        PROMPT
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )
                
                # Parse JSON
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:-3].strip()
                    
                data = json.loads(raw_text)
                results[filename] = data
                
                print(f"-> Thành công (TOTAL: {data.get('TOTAL_COST')})")
                
            except Exception as e:
                print(f"-> LỖI: {e}")
                results[filename] = {"error": str(e)}
                
        # Lưu kết quả cho folder này
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"Đã lưu kết quả của {folder} vào {output_file}")

if __name__ == "__main__":
    process_images()
