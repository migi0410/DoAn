import os
import sys
import json
import time
import re
import argparse
from PIL import Image
from difflib import SequenceMatcher
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def string_similarity(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return SequenceMatcher(None, str(a).strip().lower(), str(b).strip().lower()).ratio()

def parse_extracted_json(text):
    if not text:
        return {}
    text = text.strip()
    # Find JSON block inside markdown or raw
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r'\{.*\}', text, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return {}

def evaluate_predictions(pred, gt):
    """
    So khớp chi tiết 8 trường thông tin:
    SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT
    """
    scores = {}
    
    # 1. Single text fields
    for field in ["SELLER", "ADDRESS", "TIMESTAMP"]:
        gt_val = str(gt.get(field, "")).strip()
        pred_val = str(pred.get(field, "")).strip()
        scores[field] = round(string_similarity(pred_val, gt_val), 3)

    # 2. TOTAL_COST (so sánh số)
    gt_cost = re.sub(r'[^\d]', '', str(gt.get("TOTAL_COST", "")))
    pred_cost = re.sub(r'[^\d]', '', str(pred.get("TOTAL_COST", "")))
    scores["TOTAL_COST"] = 1.0 if (gt_cost and gt_cost == pred_cost) else round(string_similarity(pred_cost, gt_cost), 3)

    # 3. Line item fields (ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT)
    for field in ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]:
        gt_list = gt.get(field, [])
        pred_list = pred.get(field, [])
        if not isinstance(gt_list, list): gt_list = [str(gt_list)]
        if not isinstance(pred_list, list): pred_list = [str(pred_list)]
        
        gt_list = [str(x).strip() for x in gt_list if str(x).strip()]
        pred_list = [str(x).strip() for x in pred_list if str(x).strip()]

        if not gt_list and not pred_list:
            scores[field] = 1.0
            continue
        if not gt_list or not pred_list:
            scores[field] = 0.0
            continue

        # Precision & Recall matching with threshold >= 0.7 (or exact for numbers)
        matched_gt = 0
        for g in gt_list:
            best_sim = max([string_similarity(g, p) for p in pred_list], default=0.0)
            if best_sim >= 0.7:
                matched_gt += 1
        
        matched_pred = 0
        for p in pred_list:
            best_sim = max([string_similarity(p, g) for g in gt_list], default=0.0)
            if best_sim >= 0.7:
                matched_pred += 1

        rec = matched_gt / len(gt_list) if gt_list else 1.0
        prec = matched_pred / len(pred_list) if pred_list else 1.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        scores[field] = round(f1, 3)

    macro_f1 = sum(scores.values()) / len(scores)
    scores["MACRO_F1"] = round(macro_f1, 3)
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="/home/haderax/DoAn/checkpoints_qwen3_vl_qlora/final_lora_checkpoint")
    parser.add_argument("--test_file", type=str, default="/home/haderax/DoAn/official_benchmark/test_abs.jsonl")
    parser.add_argument("--output_file", type=str, default="/home/haderax/DoAn/official_benchmark/lora_eval_results.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_img_dim", type=int, default=768)
    args = parser.parse_args()

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ (EVALUATION) MÔ HÌNH QWEN3-VL-8B QLORA")
    print(f"   Adapter: {args.adapter_dir}")
    print(f"   Tập Test: {args.test_file}")
    print(f"   Số lượng mẫu: {args.max_samples}")
    print("=" * 80)

    # 1. Nạp Base Model 4-bit + LoRA Adapter
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Đang nạp base model 4-bit...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Đang ghép trọng số LoRA từ: {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256*28*28,
        max_pixels=512*28*28,
        trust_remote_code=True
    )

    # 2. Đọc tập Test
    samples = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if args.max_samples and idx >= args.max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    print(f"Đã đọc {len(samples)} mẫu test để tiến hành suy luận...")

    results = []
    prompt_text = "Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON."

    total_time = 0.0

    for idx, item in enumerate(samples):
        img_path = item["images"][0]
        gt_raw = item["messages"][1]["content"]
        gt_json = parse_extracted_json(gt_raw)

        # Đọc và resize ảnh
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > args.max_img_dim:
                scale = args.max_img_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Lỗi mở ảnh {img_path}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[img], return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.01,
                do_sample=False
            )
        elapsed = time.time() - t0
        total_time += elapsed

        pred_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_json = parse_extracted_json(pred_text)

        eval_score = evaluate_predictions(pred_json, gt_json)
        eval_score["sample_id"] = idx
        eval_score["img"] = os.path.basename(img_path)
        eval_score["latency_s"] = round(elapsed, 2)
        results.append(eval_score)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            avg_f1_so_far = sum(r["MACRO_F1"] for r in results) / len(results)
            print(f"[{idx+1}/{len(samples)}] Avg Macro-F1 hiện tại: {avg_f1_so_far*100:.2f}% (Tốc độ: {elapsed:.2f}s/mẫu)")

    # 3. Tổng kết
    field_averages = {}
    for field in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT", "MACRO_F1"]:
        field_averages[field] = round(sum(r[field] for r in results) / len(results) * 100, 2)

    avg_latency = round(total_time / len(results), 2)

    summary = {
        "num_samples": len(results),
        "avg_latency_s": avg_latency,
        "metrics": field_averages,
        "details": results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ BENCHMARK CHÍNH THỨC CỦA QWEN3-VL-8B (FINE-TUNED QLORA)")
    print("=" * 80)
    print(f"- Số lượng mẫu test: {len(results)}")
    print(f"- Tốc độ suy luận: {avg_latency}s / hóa đơn")
    print("-" * 50)
    for k, v in field_averages.items():
        print(f"  * {k:15s}: {v:.2f}%")
    print("=" * 80)
    print(f"Đã lưu chi tiết vào: {args.output_file}")

if __name__ == "__main__":
    main()
