import os
import sys
import json
import time
import base64
import io
import re
from PIL import Image
import requests
import pandas as pd
import argparse

sys.path.insert(0, '/home/haderax/DoAn/official_benchmark')
from kie_full_evaluator import evaluate_all_8_fields

BASE_DIR = '/home/haderax/DoAn/mcocr_benchmark'
TEST_JSONL = os.path.join(BASE_DIR, 'test_mcocr_gemini_8fields.jsonl')
IMG_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_CSV = os.path.join(BASE_DIR, 'mcocr_vlm_benchmark_results.csv')
OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
LORA_V2_API_URL = 'http://127.0.0.1:8000/extract'

PROMPT_SCHEMA_V2 = """Bạn là chuyên gia trích xuất thông tin hóa đơn tiếng Việt. Hãy đọc kỹ hình ảnh và trích xuất thông tin vào định dạng JSON sau:
{
  "SELLER": "Tên cửa hàng hoặc công ty",
  "ADDRESS": "Địa chỉ",
  "TIMESTAMP": "Thời gian lập hóa đơn",
  "ITEMS": [
    {
      "name": "Tên món hàng",
      "qty": "Số lượng",
      "price": "Đơn giá",
      "amount": "Thành tiền"
    }
  ],
  "TOTAL_COST": "Tổng tiền thanh toán cuối cùng"
}
Quy tắc bắt buộc:
1. Chỉ trích xuất CHÍNH XÁC những gì nhìn thấy trên ảnh. Tuyệt đối KHÔNG tự suy đoán hay bịa đặt thông tin.
2. Với từng món hàng, nhóm đủ 4 trường name, qty, price, amount vào cùng một đối tượng. Nếu hóa đơn không in đơn giá riêng, hãy để price là "".
3. Nếu trường thông tin nào không xuất hiện trên hóa đơn, hãy để giá trị là ""."""

PROMPT_SCHEMA_V1 = """Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON thuần túy.

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
}"""

def parse_json(raw_text):
    if not raw_text:
        return None
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

def normalize_items_to_flat_arrays(data):
    if not isinstance(data, dict):
        return {}
    res = dict(data)
    if "ITEMS" in res and isinstance(res["ITEMS"], list):
        if "ITEM_NAME" not in res:
            res["ITEM_NAME"] = [item.get("name", "") for item in res["ITEMS"] if isinstance(item, dict)]
            res["ITEM_QTY"] = [item.get("qty", "") for item in res["ITEMS"] if isinstance(item, dict)]
            res["ITEM_PRICE"] = [item.get("price", "") for item in res["ITEMS"] if isinstance(item, dict)]
            res["ITEM_AMOUNT"] = [item.get("amount", "") for item in res["ITEMS"] if isinstance(item, dict)]
    return res

def call_ollama(model_name, prompt, img_b64=None):
    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.0, 'num_predict': 4096, 'num_ctx': 8192}
    }
    if img_b64:
        payload['images'] = [img_b64]
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return r.json().get('response', '')
    except Exception as e:
        print(f'Error calling Ollama {model_name}: {e}')
        return ''

def call_lora_api(api_url, img_path):
    try:
        with open(img_path, 'rb') as f:
            r = requests.post(api_url, files={'file': ('img.png', f, 'image/png')}, timeout=120)
        if r.status_code == 200:
            res = r.json()
            return res.get('data', {}), res.get('latency_s', 0.0)
    except Exception as e:
        print(f'Error calling API {api_url}: {e}')
    return None, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='all', choices=['lora_v2', 'lora_v2_enhanced', 'base_v2', 'lora_v1', 'all'])
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(TEST_JSONL):
        print(f"Error: {TEST_JSONL} not found.")
        return

    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]

    if args.max_samples:
        samples = samples[:args.max_samples]

    total_samples = len(samples)
    print("=" * 80)
    print(f"🚀 MCOCR ZERO-SHOT GENERALIZATION BENCHMARK - ALL 8 FIELDS ({total_samples} SAMPLES)")
    print("=" * 80)

    existing_keys = set()
    if os.path.exists(OUTPUT_CSV):
        df_exist = pd.read_csv(OUTPUT_CSV)
        existing_keys = set(zip(df_exist['Sample_Idx'], df_exist['Model']))

    models_to_run = []
    if args.models in ['lora_v2', 'all']:
        models_to_run.append(('Qwen3-VL (8B) - LoRA v2 (Prompt v2)', 'lora_v2'))
    if args.models in ['lora_v2_enhanced', 'all_enhanced']:
        models_to_run.append(('Qwen3-VL (8B) - LoRA v2 (Enhanced)', 'lora_v2'))
    if args.models in ['base_v2', 'all']:
        models_to_run.append(('Qwen3-VL (8B) - Base (Prompt v2)', 'base_v2'))
    if args.models in ['lora_v1', 'all']:
        models_to_run.append(('Qwen3-VL (8B) - LoRA v1 (Prompt v1)', 'lora_v1'))

    for model_display_name, model_type in models_to_run:
        print(f"\n=========================================================================")
        print(f"🏁 ĐANG ĐÁNH GIÁ MCOCR: {model_display_name}")
        print(f"=========================================================================")

        hf_model = None
        hf_processor = None
        if model_type == 'lora_v1':
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            from peft import PeftModel
            from qwen_vl_utils import process_vision_info
            import torch

            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            hf_model = PeftModel.from_pretrained(base_model, "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora/final_lora_checkpoint")
            hf_model.eval()
            hf_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", min_pixels=256*28*28, max_pixels=1024*28*28, trust_remote_code=True)

        for i, s in enumerate(samples):
            fname = s['filename']
            sample_idx = s.get('Sample_Idx', i)
            if (sample_idx, model_display_name) in existing_keys:
                continue

            img_path = os.path.join(IMG_DIR, fname)
            if not os.path.exists(img_path):
                continue

            gt_json = s.get('ground_truth', {})
            gt_json = normalize_items_to_flat_arrays(gt_json)
            pred_json = None
            latency = 0.0

            if model_type == 'lora_v2':
                pred_json, latency = call_lora_api(LORA_V2_API_URL, img_path)
            elif model_type == 'base_v2':
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    w, h = pil_img.size
                    if max(w, h) > 1280:
                        scale = 1280 / max(w, h)
                        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=92)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    t0 = time.time()
                    raw = call_ollama('qwen3-vl:8b-instruct', PROMPT_SCHEMA_V2, img_b64=b64)
                    latency = round(time.time() - t0, 2)
                    pred_json = parse_json(raw)
                except Exception as e:
                    print(f"Error on {fname}: {e}")
            elif model_type == 'lora_v1':
                import torch
                from qwen_vl_utils import process_vision_info
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                if max(w, h) > 1280:
                    scale = 1280 / max(w, h)
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": PROMPT_SCHEMA_V1}]}]
                text = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = hf_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
                t0 = time.time()
                with torch.no_grad():
                    outputs = hf_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                latency = round(time.time() - t0, 2)
                out_text = hf_processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred_json = parse_json(out_text)

            pred_json = normalize_items_to_flat_arrays(pred_json)
            metrics = evaluate_all_8_fields(pred_json, gt_json)

            record = {
                'Sample_Idx': sample_idx,
                'Filename': fname,
                'Model': model_display_name,
                'Latency_s': latency,
                **metrics
            }

            df_row = pd.DataFrame([record])
            df_row.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0, index=False)
            existing_keys.add((sample_idx, model_display_name))
            print(f"[{i+1}/{total_samples}] {fname} | Latency: {latency}s | Macro-F1: {metrics['macro_f1']}% | Seller: {metrics['Seller_Sim']}% | Total: {metrics['Total_Sim']}% | Item-R: {metrics['Item_Recall']}%", flush=True)

        print(f"\n✅ Hoàn thành đánh giá MCOCR: {model_display_name}!")

if __name__ == '__main__':
    main()
