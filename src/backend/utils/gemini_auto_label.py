import os
import glob
import time
import json
import google.generativeai as genai

# ================= CẤU HÌNH =================
API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"
IMAGE_DIR = "đường_dẫn_tới_thư_mục_ảnh" # VD: "raw_data/physical_dataset/raw_images"
OUTPUT_FILE = "gemini_labels.json"
MAX_IMAGES = 100
# ============================================

# Prompt chuẩn để trích xuất thông tin KIE
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

def process_images():
    genai.configure(api_key=API_KEY)
    # Khởi tạo mô hình rẻ và nhanh nhất cho tác vụ này
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Tìm tất cả file .jpg và .png
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    # Lấy đúng 100 tấm
    image_paths = image_paths[:MAX_IMAGES]
    
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {IMAGE_DIR}")
        return

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh bằng gói Free (15 requests/phút)...")
    
    results = {}
    
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"[{i+1}/{len(image_paths)}] Đang xử lý: {filename}")
        
        try:
            # Upload ảnh lên Gemini
            myfile = genai.upload_file(path)
            
            # Gọi model
            response = model.generate_content([myfile, PROMPT])
            
            # Lấy raw text và parse JSON
            raw_text = response.text.strip()
            
            # Xử lý trường hợp model trả về block markdown code
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            data = json.loads(raw_text)
            results[filename] = data
            
            print(f"  -> Thành công: SELLER='{data.get('SELLER')}', TOTAL='{data.get('TOTAL_COST')}'")
            
            # Xóa file trên API sau khi dùng xong để dọn dẹp
            genai.delete_file(myfile.name)
            
        except Exception as e:
            print(f"  -> Lỗi: {e}")
            results[filename] = {"error": str(e)}
            
        # QUAN TRỌNG: Gói Free giới hạn 15 req/phút (~4s/req).
        # Nghỉ 4.5 giây để đảm bảo an toàn, không bị lỗi Rate Limit (429)
        time.sleep(4.5)
        
    # Lưu kết quả
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nHoàn tất! Đã lưu nhãn của {len(results)} ảnh vào {OUTPUT_FILE}")

if __name__ == "__main__":
    if API_KEY == "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY":
        print("LỖI: Vui lòng mở file script này lên và điền API_KEY của bạn vào dòng 8!")
    else:
        process_images()
