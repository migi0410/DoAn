import os
import json
import torch
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# CẤU HÌNH ĐÁNH GIÁ
# ==========================================
BASE_MODEL = "openbmb/MiniCPM-V-2_6"
LORA_PATH = "minicpmv_lora_output"  # Thư mục output mà anh/chị chọn trên WebUI
TEST_JSONL = "test.jsonl"  # File chứa 500 ảnh test của anh/chị
# ==========================================

def parse_json_output(text):
    """Trích xuất chuỗi JSON từ câu trả lời của mô hình"""
    # Tìm đoạn chữ nằm trong dấu ngoặc nhọn {}
    match = re.search(r'\{.*?\}', text.replace('\n', ' '), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return {}
    return {}

def calculate_f1(pred, true):
    """Tính điểm F1 (Độ chuẩn xác) cho từng trường thông tin"""
    fields = ['company', 'address', 'date', 'total']
    scores = {}
    for f in fields:
        p = str(pred.get(f, "")).strip().lower()
        t = str(true.get(f, "")).strip().lower()
        
        # Nếu khớp hoàn toàn 100%
        if p == t and t != "":
            scores[f] = 1.0
        # Nếu nhãn gốc bị rỗng
        elif t == "":
            scores[f] = 1.0 if p == "" else 0.0
        else:
            # Tính độ khớp theo từng chữ (Token-level F1)
            p_tokens = set(p.split())
            t_tokens = set(t.split())
            if not t_tokens:
                scores[f] = 0.0
                continue
            common = p_tokens.intersection(t_tokens)
            if not common:
                scores[f] = 0.0
                continue
            precision = len(common) / len(p_tokens)
            recall = len(common) / len(t_tokens)
            scores[f] = 2 * (precision * recall) / (precision + recall)
    return scores

def main():
    if not os.path.exists(LORA_PATH):
        print(f"❌ Lỗi: Không tìm thấy thư mục LoRA '{LORA_PATH}'. Anh/chị đã train xong chưa?")
        return
        
    print("⏳ Bước 1: Đang tải mô hình gốc MiniCPM-V 2.6 (Sẽ hơi lâu xíu)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("⏳ Bước 2: Đang lắp não (LoRA Adapter) vừa train vào mô hình...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    print("⏳ Bước 3: Đọc danh sách 500 hóa đơn Test...")
    if not os.path.exists(TEST_JSONL):
        print(f"❌ Lỗi: Không tìm thấy file {TEST_JSONL}. Anh/chị hãy tạo file test.jsonl chứa 500 ảnh nhé.")
        return
        
    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    total_scores = {'company': 0, 'address': 0, 'date': 0, 'total': 0}
    count = 0
    
    print(f"🚀 Bắt đầu hành trình chấm điểm {len(lines)} hóa đơn (Quá trình này tự động 100%)...")
    for line in tqdm(lines):
        data = json.loads(line)
        img_path = data['images'][0]
        
        # Lấy đáp án chuẩn (Ground Truth) từ file JSONL
        ground_truth_str = ""
        user_prompt = ""
        for msg in data['messages']:
            if msg['role'] == 'user':
                user_prompt = msg['content']
            elif msg['role'] == 'assistant':
                ground_truth_str = msg['content']
        
        # Xóa tag hình ảnh <image> trong prompt của user (nếu có)
        user_prompt = user_prompt.replace("<image>", "").strip()
        if not user_prompt:
            user_prompt = "Trích xuất thông tin công ty, địa chỉ, ngày, tổng tiền từ hóa đơn này thành JSON."
            
        true_json = parse_json_output(ground_truth_str)
        
        # AI tự làm bài thi
        try:
            image = Image.open(img_path).convert('RGB')
            # Cấu trúc prompt đặc biệt của MiniCPM-V 2.6
            msgs = [{'role': 'user', 'content': [image, user_prompt]}]
            
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False, # Dùng chế độ Greedy để xuất đáp án chính xác nhất, không sáng tạo lung tung
            )
            
            pred_json = parse_json_output(res)
            
            # Tính điểm ngay lập tức
            scores = calculate_f1(pred_json, true_json)
            for k in total_scores:
                total_scores[k] += scores.get(k, 0.0)
            count += 1
            
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {img_path}: {e}")
            
    # Hồi hộp chờ kết quả
    if count == 0:
        print("❌ Không chấm được bài nào, vui lòng kiểm tra lại file test.jsonl")
        return
        
    print("\n" + "★"*50)
    print("🏆 BẢNG ĐIỂM TỔNG KẾT (F1-SCORE) 🏆")
    print("Dùng bảng điểm này copy thẳng vào PowerPoint báo cáo Đồ Án:")
    print("★"*50)
    print(f"📸 Số lượng hóa đơn đã test: {count}/{len(lines)}")
    print(f"🏢 Tên Công Ty (Company) : {total_scores['company'] / count * 100:.2f} %")
    print(f"📍 Địa Chỉ (Address)     : {total_scores['address'] / count * 100:.2f} %")
    print(f"📅 Ngày Tháng (Date)     : {total_scores['date'] / count * 100:.2f} %")
    print(f"💵 Tổng Tiền (Total)     : {total_scores['total'] / count * 100:.2f} %")
    print("★"*50)

if __name__ == "__main__":
    main()
