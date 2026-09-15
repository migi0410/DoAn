import os
import sys
import json
import time
import base64
import re
import cv2
import requests
import pandas as pd

sys.path.insert(0, '/home/haderax/DoAn/official_benchmark')
from kie_full_evaluator import evaluate_all_8_fields

BASE_DIR = '/home/haderax/DoAn/official_benchmark'
TEST_JSONL = os.path.join(BASE_DIR, 'test.jsonl')
IMG_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_CSV = os.path.join(BASE_DIR, 'benchmark_results.csv')
OLLAMA_URL = 'http://localhost:11434/api/generate'

def parse_deepseek_markdown_to_json(text):
    res = {
        'SELLER': '',
        'ADDRESS': '',
        'TIMESTAMP': '',
        'ITEM_NAME': [],
        'ITEM_QTY': [],
        'ITEM_PRICE': [],
        'ITEM_AMOUNT': [],
        'TOTAL_COST': ''
    }
    clean_lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not clean_lines:
        return res
        
    for l in clean_lines[:5]:
        l_clean = re.sub(r'^[#*\-\s]+|[#*\-\s]+$', '', l)
        if len(l_clean) > 2 and not any(k in l_clean.lower() for k in ['hóa đơn', 'hòa đơn', 'phiếu', 'ngày', 'đt', 'sđt', 'so ', 'số ', 'hd số', 'terminal', 'partner', 'coffee']):
            if 'starbucks' in l_clean.lower():
                for l_sub in clean_lines[:6]:
                    if 'starbucks' in l_sub.lower() and len(l_sub) > len(l_clean):
                        l_clean = re.sub(r'^[#*\-\s]+|[#*\-\s]+$', '', l_sub)
                        break
            res['SELLER'] = l_clean
            break

    for l in clean_lines[:8]:
        l_clean = re.sub(r'^[#*\-\s]+|[#*\-\s]+$', '', l)
        if any(k in l_clean.lower() for k in ['số ', 'so ', 'đường ', 'quan ', 'quận ', 'phường ', 'da nang', 'đà nẵng', 'hải phòng', 'đồng nai', 'đống nai', 'tp', 'hà nội']):
            res['ADDRESS'] = l_clean
            break

    time_m = re.search(r'(?:Date[:\s]*|Thời gian[:\s]*|Ngày mua[:\s]*)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?)', text)
    if time_m:
        res['TIMESTAMP'] = time_m.group(1).strip()

    total_candidates = []
    for i, l in enumerate(clean_lines):
        l_no_md = re.sub(r'[*_#]+', '', l)
        l_lower = l_no_md.lower()
        if any(k in l_lower for k in ['tổng tiền', 'thành tiền', 'tổng cộng', 'total', 'tiền mặt', 'thanh toán']) and not 'tiền hàng' in l_lower and not 'thuế vat' in l_lower:
            m = re.findall(r'[\d,.]+', l_no_md)
            digits = [d for d in m if any(c.isdigit() for c in d) and len(d) >= 3]
            if digits:
                total_candidates.append(digits[-1])
            elif i + 1 < len(clean_lines):
                next_l = re.sub(r'[*_#]+', '', clean_lines[i+1])
                m_next = re.findall(r'[\d,.]+', next_l)
                digits_next = [d for d in m_next if any(c.isdigit() for c in d) and len(d) >= 3]
                if digits_next:
                    total_candidates.append(digits_next[-1])
                
    if total_candidates:
        res['TOTAL_COST'] = total_candidates[-1]

    has_table = any(l.startswith('|') and not l.startswith('|---') for l in clean_lines)
    if has_table:
        for line in clean_lines:
            if line.startswith('|') and not line.startswith('|---') and not any(k in line.lower() for k in ['sản phẩm', 'tên món', 'món ăn', 'sl', 'tổng cộng', 'item']):
                parts = [re.sub(r'[*_]+', '', p.strip()) for p in line.split('|')[1:-1] if p.strip()]
                if len(parts) >= 2:
                    name = parts[0]
                    amount = parts[-1]
                    qty, price = '1', amount
                    m_calc = re.search(r'(\d+)\s*(?:\\times|x|\*)\s*([\d,.]+)', line)
                    if m_calc:
                        qty = m_calc.group(1)
                        price = m_calc.group(2)
                    elif len(parts) >= 4:
                        qty = parts[1]
                        price = parts[2]
                    res['ITEM_NAME'].append(name)
                    res['ITEM_QTY'].append(qty)
                    res['ITEM_PRICE'].append(price)
                    res['ITEM_AMOUNT'].append(amount)
    else:
        for i in range(len(clean_lines) - 2):
            l1, l2, l3 = clean_lines[i], clean_lines[i+1], clean_lines[i+2]
            is_qty = re.match(r'^\d+(\.\d+)?$', l2)
            is_amount = re.match(r'^[\d,.]+$', l3) and any(c in l3 for c in [',', '.'])
            is_name = len(l1) > 2 and not any(k in l1.lower() for k in ['subtotal', 'total', 'cash', 'date', 'terminal', 'partner', 'item', 'qty', 'amount'])
            
            if is_qty and is_amount and is_name:
                res['ITEM_NAME'].append(l1)
                res['ITEM_QTY'].append(str(int(float(l2))))
                total_val = float(l3.replace(',', '').replace('.', ''))
                qty_val = float(l2)
                unit_price = str(int(total_val / qty_val)) if qty_val > 0 else l3
                res['ITEM_PRICE'].append(unit_price)
                res['ITEM_AMOUNT'].append(l3)

    return res

def call_ollama(model_name, prompt, img_b64=None, force_json=False):
    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.0, 'num_predict': 4096, 'num_ctx': 8192}
    }
    if img_b64:
        payload['images'] = [img_b64]
    if force_json:
        payload['format'] = 'json'
        
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return r.json().get('response', '')
    except Exception as e:
        print(f'Error calling {model_name}: {e}')
        return ''

def parse_json(raw_text):
    if not raw_text:
        return None
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

def main():
    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    num_to_run = len(all_lines)

    print('=' * 105, flush=True)
    print(f'🚀 CHẠY BENCHMARK NATIVE NỘI BỘ TRÊN POPOS ({num_to_run} MẪU)', flush=True)
    print(f'📁 File CSV kết quả: {OUTPUT_CSV}', flush=True)
    print('=' * 105, flush=True)

    records = []
    completed_filenames = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
            records = df_old.to_dict('records')
            completed_filenames = set(df_old['Filename'].unique())
            print(f'🔄 Đã tìm thấy {len(completed_filenames)} mẫu trong CSV. Tiếp tục chạy mẫu tiếp theo!', flush=True)
        except Exception:
            pass

    vlm_prompt = """Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON thuần túy.

Cấu trúc JSON yêu cầu:
{
  "SELLER": "Tên cửa hàng",
  "ADDRESS": "Địa chỉ",
  "TIMESTAMP": "Thời gian",
  "ITEM_NAME": ["Tên món 1", "Tên món 2"],
  "ITEM_QTY": ["SL 1", "SL 2"],
  "ITEM_PRICE": ["Đơn giá 1", "Đơn giá 2"],
  "ITEM_AMOUNT": ["Thành tiền 1", "Thành tiền 2"],
  "TOTAL_COST": "Tổng tiền thanh toán"
}
"""
    ocr_prompt = 'Read all text in this image.'

    for i in range(num_to_run):
        sample = json.loads(all_lines[i])
        fname = os.path.basename(sample['images'][0])
        if fname in completed_filenames:
            continue

        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            print(f'⚠️ Không tìm thấy ảnh: {img_path}', flush=True)
            continue
            
        gt_json = parse_json(sample['messages'][1]['content']) or {}
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        b64 = base64.b64encode(buf).decode('utf-8')
        
        print(f'\n[{i+1}/{num_to_run}] 📸 Đang xử lý: {fname}', flush=True)

        # 1. Qwen3-VL (8B)
        t0 = time.time()
        qwen_raw = call_ollama('qwen3-vl:8b-instruct', vlm_prompt, img_b64=b64)
        qwen_time = time.time() - t0
        qwen_json = parse_json(qwen_raw)
        qwen_m = evaluate_all_8_fields(qwen_json, gt_json)
        print(f"  👑 Qwen3-VL: {qwen_time:.2f}s | Macro-F1: {qwen_m['macro_f1']}% | Seller-Sim: {qwen_m['Seller_Sim']}% | Total-Sim: {qwen_m['Total_Sim']}% | Item-Recall: {qwen_m['Item_Recall']}%", flush=True)
        records.append({
            'Sample_Idx': i, 'Filename': fname, 'Model': 'Qwen3-VL (8B)',
            'Latency_s': round(qwen_time, 2), **qwen_m
        })

        # 2. MiniCPM-V (8B)
        t0 = time.time()
        cpm_raw = call_ollama('minicpm-v:latest', vlm_prompt, img_b64=b64, force_json=True)
        cpm_time = time.time() - t0
        cpm_json = parse_json(cpm_raw)
        cpm_m = evaluate_all_8_fields(cpm_json, gt_json)
        print(f"  🥈 MiniCPM-V: {cpm_time:.2f}s | Macro-F1: {cpm_m['macro_f1']}% | Seller-Sim: {cpm_m['Seller_Sim']}% | Total-Sim: {cpm_m['Total_Sim']}% | Item-Recall: {cpm_m['Item_Recall']}%", flush=True)
        records.append({
            'Sample_Idx': i, 'Filename': fname, 'Model': 'MiniCPM-V (8B)',
            'Latency_s': round(cpm_time, 2), **cpm_m
        })

        # 3. DeepSeek-OCR
        t0 = time.time()
        ds_raw = call_ollama('deepseek-ocr:latest', ocr_prompt, img_b64=b64)
        ds_ocr_time = time.time() - t0
        
        # 3a. DeepSeek + Regex
        t_reg = time.time()
        ds_reg_json = parse_deepseek_markdown_to_json(ds_raw)
        reg_time = time.time() - t_reg
        reg_m = evaluate_all_8_fields(ds_reg_json, gt_json)
        total_ds_reg_time = ds_ocr_time + reg_time
        print(f"  ⚡ DeepSeek + Regex: {total_ds_reg_time:.2f}s | Macro-F1: {reg_m['macro_f1']}% | Seller-Sim: {reg_m['Seller_Sim']}% | Total-Sim: {reg_m['Total_Sim']}% | Item-Recall: {reg_m['Item_Recall']}%", flush=True)
        records.append({
            'Sample_Idx': i, 'Filename': fname, 'Model': 'DeepSeek + Regex',
            'Latency_s': round(total_ds_reg_time, 2), **reg_m
        })

        # 3b. DeepSeek + Qwen2.5-7B
        t_llm0 = time.time()
        llm_prompt = f"""Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ văn bản hóa đơn sau dưới dạng JSON thuần túy:

{ds_raw}

JSON schema:
{{
  "SELLER": "...",
  "ADDRESS": "...",
  "TIMESTAMP": "...",
  "ITEM_NAME": ["..."],
  "ITEM_QTY": ["..."],
  "ITEM_PRICE": ["..."],
  "ITEM_AMOUNT": ["..."],
  "TOTAL_COST": "..."
}}
"""
        llm_raw = call_ollama('qwen2.5:7b', llm_prompt)
        llm_time = time.time() - t_llm0
        ds_llm_json = parse_json(llm_raw)
        llm_m = evaluate_all_8_fields(ds_llm_json, gt_json)
        total_ds_llm_time = ds_ocr_time + llm_time
        print(f"  🧠 DeepSeek + Qwen2.5: {total_ds_llm_time:.2f}s | Macro-F1: {llm_m['macro_f1']}% | Seller-Sim: {llm_m['Seller_Sim']}% | Total-Sim: {llm_m['Total_Sim']}% | Item-Recall: {llm_m['Item_Recall']}%", flush=True)
        records.append({
            'Sample_Idx': i, 'Filename': fname, 'Model': 'DeepSeek + Qwen2.5',
            'Latency_s': round(total_ds_llm_time, 2), **llm_m
        })
        
        # Incremental save
        df_curr = pd.DataFrame(records)
        df_curr.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 115, flush=True)
    print('🏆 BẢNG TỔNG HỢP TOÀN BỘ 1166 MẪU TRÊN POPOS:', flush=True)
    print('=' * 115, flush=True)
    summary = df_curr.groupby('Model').agg({
        'Latency_s': 'mean',
        'valid_json': 'mean',
        'Seller_Sim': 'mean',
        'Total_Sim': 'mean',
        'Item_Recall': 'mean',
        'macro_f1': 'mean',
        'micro_f1': 'mean',
        'exact_match': 'mean'
    }).reset_index()
    summary['Valid_JSON (%)'] = (summary['valid_json'] * 100).round(1)
    summary['Seller_Sim (%)'] = summary['Seller_Sim'].round(1)
    summary['Total_Sim (%)'] = summary['Total_Sim'].round(1)
    summary['Item_Recall (%)'] = summary['Item_Recall'].round(1)
    summary['Macro_F1 (%)'] = summary['macro_f1'].round(1)
    summary['Micro_F1 (%)'] = summary['micro_f1'].round(1)
    summary['Exact_Match (%)'] = summary['exact_match'].round(1)
    summary['Latency_s'] = summary['Latency_s'].round(2)
    summary1 = summary[['Model', 'Latency_s', 'Valid_JSON (%)', 'Seller_Sim (%)', 'Total_Sim (%)', 'Item_Recall (%)', 'Macro_F1 (%)', 'Micro_F1 (%)', 'Exact_Match (%)']]
    print(summary1.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
