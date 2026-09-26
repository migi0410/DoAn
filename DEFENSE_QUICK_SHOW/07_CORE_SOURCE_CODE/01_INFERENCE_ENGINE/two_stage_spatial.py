# -*- coding: utf-8 -*-
"""
AVIR-KIE Two-Stage Hybrid Extraction Pipeline
Architecture: PaddleOCR Bounding Box Detection + Spatial Line Grouping + Qwen2.5 (7B) Structured KIE

Purpose:
Ablation & comparative study baseline evaluating the trade-offs between
End-to-End Multimodal Vision-Language Models (Qwen3-VL 8B LoRA v2) and
Two-Stage Hybrid Systems with geometric spatial line grouping.
"""

import os
import re
import sys
import json
import time
import subprocess
import requests
from typing import Dict, Any, List, Tuple
from PIL import Image

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
PADDLE_ENV_PYTHON = os.environ.get("PADDLE_ENV_PYTHON", "/home/haderax/paddle_env/bin/python")

def get_paddle_ocr_boxes(image_path: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Trích xuất danh sách Bounding Boxes từ ảnh hóa đơn sử dụng PaddleOCR (PP-OCR).
    Đảm bảo cô lập môi trường để không gây xung đột dependency với PyTorch/CUDA.
    """
    t0 = time.time()
    
    # 1. Thử import trực tiếp nếu môi trường hiện tại có sẵn paddleocr
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=False, lang="vi", show_log=False)
        raw_res = ocr.ocr(image_path, cls=False)
        boxes = []
        if raw_res and raw_res[0]:
            for line in raw_res[0]:
                box, (text, score) = line
                boxes.append({"box": box, "text": text, "score": float(score)})
        return boxes, round(time.time() - t0, 3)
    except ImportError:
        pass

    # 2. Gọi qua môi trường paddle_env cô lập
    python_bin = PADDLE_ENV_PYTHON if os.path.exists(PADDLE_ENV_PYTHON) else sys.executable
    cmd = [
        python_bin, "-c",
        f"""
import json
import sys
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='vi', show_log=False)
    res = ocr.ocr(r'{image_path}', cls=False)
    out = []
    if res and res[0]:
        for line in res[0]:
            box, (text, score) = line
            out.append({{'box': box, 'text': text, 'score': float(score)}})
    print('__OCR_JSON_START__' + json.dumps(out) + '__OCR_JSON_END__')
except Exception as e:
    print('__OCR_ERROR__' + str(e), file=sys.stderr)
"""
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
        stdout = proc.stdout
        if "__OCR_JSON_START__" in stdout:
            json_str = stdout.split("__OCR_JSON_START__")[1].split("__OCR_JSON_END__")[0]
            boxes = json.loads(json_str)
            return boxes, round(time.time() - t0, 3)
        else:
            print(f"⚠️ [PaddleOCR Subprocess Warning]: {proc.stderr}")
    except Exception as e:
        print(f"⚠️ [PaddleOCR Invocation Error]: {e}")

    return [], round(time.time() - t0, 3)

def run_two_stage_pipeline(image_path: str, ollama_url: str = OLLAMA_URL) -> Dict[str, Any]:
    """
    Thực thi toàn diện quy trình trích xuất thông tin hóa đơn 2 tầng:
    Stage 1: PaddleOCR phát hiện hộp tọa độ tứ giác [[x1,y1]..[x4,y4]].
    Stage 2: Spatial Line Grouping gom cụm dòng theo trục Y và phân biệt cột trái/phải.
             Tự động phát hiện và gộp dòng mô tả phụ / topping không có tiền vào món chính phía trên.
    Stage 3: Qwen2.5 (7B) cấu trúc hóa thành lược đồ KIE chuẩn.
    """
    t_total_start = time.time()
    
    # -------------------------------------------------------------
    # STAGE 1: PaddleOCR Bounding Box Detection
    # -------------------------------------------------------------
    boxes_raw, ocr_latency = get_paddle_ocr_boxes(image_path)
    if not boxes_raw:
        return {
            "success": False,
            "error": "PaddleOCR không phát hiện được chữ hoặc gặp lỗi khởi tạo",
            "latency_s": round(time.time() - t_total_start, 2),
            "model_version": "two_stage_spatial"
        }

    img = Image.open(image_path)
    img_w, img_h = img.size

    norm_boxes = []
    for item in boxes_raw:
        b = item["box"]
        text = str(item["text"]).strip()
        score = float(item["score"])
        if not text:
            continue
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        norm_boxes.append({
            "ymin": ymin, "ymax": ymax,
            "xmin": xmin, "xmax": xmax,
            "ycenter": (ymin + ymax) / 2.0,
            "xcenter": (xmin + xmax) / 2.0,
            "h": max(ymax - ymin, 1),
            "text": text,
            "score": score
        })

    norm_boxes.sort(key=lambda x: (x["ymin"], x["xmin"]))

    # -------------------------------------------------------------
    # STAGE 2: Spatial Line Clustering & Column Projection
    # -------------------------------------------------------------
    t_spatial_start = time.time()
    lines = []
    for b in norm_boxes:
        matched_line = None
        for line in lines:
            line_ycenter = line["ycenter"]
            line_h = line["h"]
            # Kiểm tra độ lệch tâm đứng trong phạm vi dải ngang
            if abs(b["ycenter"] - line_ycenter) < max(line_h, b["h"]) * 0.60:
                matched_line = line
                break
        if matched_line:
            matched_line["boxes"].append(b)
            matched_line["ymin"] = min(matched_line["ymin"], b["ymin"])
            matched_line["ymax"] = max(matched_line["ymax"], b["ymax"])
            matched_line["ycenter"] = (matched_line["ymin"] + matched_line["ymax"]) / 2.0
            matched_line["h"] = matched_line["ymax"] - matched_line["ymin"]
        else:
            lines.append({
                "ymin": b["ymin"],
                "ymax": b["ymax"],
                "ycenter": b["ycenter"],
                "h": b["h"],
                "boxes": [b]
            })

    lines.sort(key=lambda l: l["ymin"])

    # Phân tách cột bên trái (Tên món, SL) và cột bên phải (Thành tiền)
    amount_x_threshold = img_w * 0.60
    currency_pattern = re.compile(r'\d{1,3}(?:[.,]\d{3})+|\b0\b')

    structured_lines = []
    for l in lines:
        l["boxes"].sort(key=lambda b: b["xmin"])
        left_boxes = [b for b in l["boxes"] if b["xcenter"] < amount_x_threshold]
        right_boxes = [b for b in l["boxes"] if b["xcenter"] >= amount_x_threshold]

        left_text = " ".join(b["text"] for b in left_boxes).strip()
        right_text = " ".join(b["text"] for b in right_boxes).strip()

        # Kiểm tra xem phía bên phải có số tiền không
        has_amount = bool(currency_pattern.search(right_text))

        # Xử lý trường hợp đặc biệt: 1 box duy nhất chứa cả tên món và số tiền ở đuôi
        if not right_boxes and left_boxes:
            last_b = left_boxes[-1]
            if last_b["xmin"] > img_w * 0.50 and currency_pattern.search(last_b["text"]):
                right_text = last_b["text"]
                left_text = " ".join(b["text"] for b in left_boxes[:-1]).strip()
                has_amount = True

        structured_lines.append({
            "ymin": l["ymin"],
            "ymax": l["ymax"],
            "left": left_text,
            "right": right_text,
            "has_amount": has_amount,
            "full_text": " ".join(b["text"] for b in l["boxes"]).strip()
        })

    # Xây dựng biểu diễn văn bản có phân cấp cấu trúc không gian
    summary_keywords = ["tổng", "tong", "thành tiền", "thanh tien", "total", "tiền mặt", "tien mat", "momo"]
    representation_lines = []

    for idx, sl in enumerate(structured_lines):
        full_lower = sl["full_text"].lower()
        is_summary = any(k in full_lower for k in summary_keywords)
        
        if is_summary:
            representation_lines.append(f"Line {idx+1} [TỔNG CỘNG / THANH TOÁN]: {sl['full_text']}")
        elif sl["has_amount"] and sl["left"]:
            representation_lines.append(f"Line {idx+1} [MÓN HÀNG]: {sl['left']} ===> Thành tiền: {sl['right']}")
        elif not sl["has_amount"] and sl["left"]:
            representation_lines.append(f"Line {idx+1} [DÒNG MÔ TẢ PHỤ / TOPPING - KHÔNG CÓ TIỀN]: {sl['left']}")
        else:
            representation_lines.append(f"Line {idx+1}: {sl['full_text']}")

    prompt_content = "\n".join(representation_lines)
    spatial_latency = round(time.time() - t_spatial_start, 3)

    # -------------------------------------------------------------
    # STAGE 3: Qwen2.5 (7B) Structured JSON Parsing
    # -------------------------------------------------------------
    t_llm_start = time.time()
    prompt = f"""Dưới đây là các dòng văn bản từ hóa đơn đã được PaddleOCR bóc tách tọa độ không gian (Spatial Grouping):
{prompt_content}

QUY TẮC BÓC TÁCH:
1. SELLER: Tên thương hiệu/cửa hàng (ví dụ: HIGHLANDS COFFEE).
2. ADDRESS: Địa chỉ cửa hàng.
3. TIMESTAMP: Ngày giờ xuất hóa đơn (DD-MM-YYYY HH:MM).
4. ITEMS: Danh sách các món hàng mua:
   - Các dòng ghi [DÒNG MÔ TẢ PHỤ / TOPPING - KHÔNG CÓ TIỀN] nằm ngay sau một món hàng là dòng mô tả phụ / topping (ví dụ: Com va Kem La Dua L, kem cheese...). BẮT BUỘC gộp dòng phụ này vào tên của món chính ngay phía trên!
   - Không được tạo món riêng cho dòng phụ không có tiền.
   - Món nào có thành tiền (kể cả 0 đ) là một món hoàn chỉnh.
5. TOTAL_COST: Tổng số tiền thanh toán thực tế cuối cùng.

ĐỊNH DẠNG JSON DUY NHẤT:
{{
  "SELLER": "...",
  "ADDRESS": "...",
  "TIMESTAMP": "...",
  "ITEMS": [
    {{"name": "...", "qty": "1", "price": "...", "amount": "..."}}
  ],
  "TOTAL_COST": "..."
}}
Chỉ xuất mã JSON, không kèm bất kỳ giải thích nào."""

    raw_response = ""
    parsed = {}
    llm_latency = 0.0

    try:
        payload = {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0.0, "num_predict": 1024}
        }
        resp = requests.post(ollama_url, json=payload, timeout=60)
        llm_latency = round(time.time() - t_llm_start, 2)
        if resp.status_code == 200:
            raw_response = resp.json().get("response", "")
            m = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
    except Exception as e_llm:
        print(f"⚠️ [Qwen2.5 Ollama Error]: {e_llm}")

    # Fallback dự phòng nếu LLM thất bại: Bóc tách trực tiếp từ cấu trúc không gian
    if not parsed:
        parsed = fallback_spatial_regex(structured_lines)
        raw_response = json.dumps(parsed, ensure_ascii=False, indent=2)

    # Chuẩn hóa ITEMS và danh sách tương ứng
    if "ITEMS" in parsed and isinstance(parsed["ITEMS"], list):
        clean_items = []
        for it in parsed["ITEMS"]:
            if isinstance(it, dict):
                clean_items.append({
                    "name": str(it.get("name", "")),
                    "qty": str(it.get("qty", "1")),
                    "price": str(it.get("price", it.get("amount", ""))),
                    "amount": str(it.get("amount", ""))
                })
        parsed["ITEMS"] = clean_items
        parsed["ITEM_NAME"] = [it["name"] for it in clean_items]
        parsed["ITEM_QTY"] = [it["qty"] for it in clean_items]
        parsed["ITEM_PRICE"] = [it["price"] for it in clean_items]
        parsed["ITEM_AMOUNT"] = [it["amount"] for it in clean_items]

    total_latency = round(time.time() - t_total_start, 2)
    return {
        "success": bool(parsed),
        "data": parsed,
        "raw": raw_response,
        "latency_breakdown": {
            "ocr_s": ocr_latency,
            "spatial_s": spatial_latency,
            "llm_s": llm_latency,
            "total_s": total_latency
        },
        "latency_s": total_latency,
        "model_version": "two_stage_spatial"
    }

def fallback_spatial_regex(structured_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bộ bóc tách toán tử hình học dự phòng khi LLM ngoại tuyến."""
    seller = structured_lines[0]["full_text"] if structured_lines else "Cửa hàng"
    address = ""
    timestamp = ""
    total = ""
    items = []

    for sl in structured_lines:
        txt = sl["full_text"]
        if not address and any(k in txt.lower() for k in ["đ/c", "địa chỉ", "đường", "quận", "tp", "phường"]):
            address = txt
        if not timestamp:
            m = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', txt)
            if m:
                timestamp = txt
        if any(k in txt.lower() for k in ["tổng", "thành tiền", "total"]):
            nums = re.findall(r'\d{1,3}(?:[.,]\d{3})+', txt)
            if nums:
                total = nums[-1]

    # Duyệt các dòng có thành tiền và gộp dòng mô tả
    prev_item = None
    for sl in structured_lines:
        if sl["has_amount"] and sl["left"]:
            item = {
                "name": sl["left"],
                "qty": "1",
                "price": sl["right"],
                "amount": sl["right"]
            }
            items.append(item)
            prev_item = item
        elif not sl["has_amount"] and sl["left"] and prev_item:
            # Gộp vào món trước đó
            prev_item["name"] += f" ({sl['left']})"

    return {
        "SELLER": seller,
        "ADDRESS": address,
        "TIMESTAMP": timestamp,
        "ITEMS": items,
        "TOTAL_COST": total
    }
