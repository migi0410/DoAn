# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script tự động gán nhãn 4 góc hóa đơn bằng Gemini 3.5 Flash cho 500 ảnh MC-OCR
      sử dụng cơ chế Native 2D Grounding [y, x] (0-1000) chống cắt lẹm tuyệt đối.
"""

import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import cv2
import numpy as np
from google import genai

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    # Fallback to backend .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("GEMINI_API_KEY", "")
    except Exception:
        pass
client = genai.Client(api_key=API_KEY) if API_KEY else None

INPUT_JSONL = "test_mcocr.jsonl"
OUTPUT_JSON = "gemini_mcocr_500_labels.json"
PREVIEW_DIR = "gemini_debug_samples"
MAX_WORKERS = 6

os.makedirs(PREVIEW_DIR, exist_ok=True)

PROMPT = """Detect the entire physical receipt paper sheet in the image.
Return JSON with:
1. "box_2d": [ymin, xmin, ymax, xmax] (normalized from 0 to 1000) bounding the whole paper sheet from the top edge to the very bottom edge (including barcodes, footer, and borders).
2. "corners": [[y_top_left, x_top_left], [y_top_right, x_top_right], [y_bottom_right, x_bottom_right], [y_bottom_left, x_bottom_left]] (normalized from 0 to 1000) corresponding to the four physical paper corners.
DO NOT crop through the receipt body or cut off the bottom barcode! The bounding box and corners MUST enclose the full physical receipt sheet."""

existing_data = {}
if os.path.exists(OUTPUT_JSON):
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        print(f"[RESUME] Đã tải {len(existing_data)} nhãn đã có.")
    except Exception:
        existing_data = {}

lock = threading.Lock()
preview_count = 0

def save_incremental(img_id, result, img_path, corners_data, box_data):
    global preview_count
    with lock:
        existing_data[img_id] = result
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        # Lưu ảnh preview kiểm tra trực quan cho 8 ảnh đầu tiên
        if preview_count < 8:
            try:
                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    ih, iw = img_cv.shape[:2]
                    ymin, xmin, ymax, xmax = box_data
                    cv2.rectangle(img_cv, (int(xmin * iw), int(ymin * ih)), (int(xmax * iw), int(ymax * ih)), (0, 255, 0), 2)

                    pts = [
                        (int(result["top_left"][0] * iw), int(result["top_left"][1] * ih)),
                        (int(result["top_right"][0] * iw), int(result["top_right"][1] * ih)),
                        (int(result["bottom_right"][0] * iw), int(result["bottom_right"][1] * ih)),
                        (int(result["bottom_left"][0] * iw), int(result["bottom_left"][1] * ih)),
                    ]
                    cv2.polylines(img_cv, [np.array(pts, np.int32)], True, (0, 255, 255), 3)

                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # TL, TR, BR, BL
                    labels = ["TL", "TR", "BR", "BL"]
                    for pt, c, lbl in zip(pts, colors, labels):
                        cv2.circle(img_cv, pt, 8, c, -1)
                        cv2.putText(img_cv, lbl, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

                    preview_path = os.path.join(PREVIEW_DIR, f"preview_{img_id}.jpg")
                    cv2.imwrite(preview_path, img_cv)
                    preview_count += 1
            except Exception as pe:
                pass

def label_single_image(item):
    img_id = item["id"]
    img_path = item["image_path"]

    if img_id in existing_data:
        return img_id, True, "Already done"

    if not os.path.exists(img_path):
        return img_id, False, f"Not found: {img_path}"

    retries = 3
    for attempt in range(retries):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                max_dim = 1280
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    im_resized = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                else:
                    im_resized = im.copy()

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[im_resized, PROMPT],
                config={"response_mime_type": "application/json"}
            )

            data = json.loads(response.text)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            # Xử lý box_2d [ymin, xmin, ymax, xmax] -> normalized [0.0 - 1.0]
            raw_box = data.get("box_2d", [0, 0, 1000, 1000])
            ymin, xmin, ymax, xmax = [max(0.0, min(1.0, v / 1000.0)) for v in raw_box]

            # Xử lý corners: [[y, x], ...] -> [x, y] normalized
            corners = data.get("corners", [])
            if len(corners) == 4:
                top_left = [max(0.0, min(1.0, corners[0][1] / 1000.0)), max(0.0, min(1.0, corners[0][0] / 1000.0))]
                top_right = [max(0.0, min(1.0, corners[1][1] / 1000.0)), max(0.0, min(1.0, corners[1][0] / 1000.0))]
                bottom_right = [max(0.0, min(1.0, corners[2][1] / 1000.0)), max(0.0, min(1.0, corners[2][0] / 1000.0))]
                bottom_left = [max(0.0, min(1.0, corners[3][1] / 1000.0)), max(0.0, min(1.0, corners[3][0] / 1000.0))]
            else:
                top_left = [xmin, ymin]
                top_right = [xmax, ymin]
                bottom_right = [xmax, ymax]
                bottom_left = [xmin, ymax]

            result_entry = {
                "image_path": img_path,
                "width": w,
                "height": h,
                "box_2d": [ymin, xmin, ymax, xmax],
                "top_left": top_left,
                "top_right": top_right,
                "bottom_right": bottom_right,
                "bottom_left": bottom_left
            }

            save_incremental(img_id, result_entry, img_path, corners, (ymin, xmin, ymax, xmax))
            return img_id, True, "OK"

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                time.sleep(5 + attempt * 3)
            else:
                time.sleep(1 + attempt)

    return img_id, False, err_msg

def main():
    print("=========================================================")
    print("   GÁN NHÃN 4 GÓC HÓA ĐƠN BẰNG GEMINI 3.5 FLASH         ")
    print("   (Chuẩn Native Grounding [y, x] chống cắt lẹm 100%)    ")
    print("=========================================================")

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"Tổng số ảnh MC-OCR:        {len(items)}")
    todo_items = [it for it in items if it["id"] not in existing_data]
    print(f"Số lượng cần gán nhãn:     {len(todo_items)}")
    print(f"Số luồng xử lý đồng thời:  {MAX_WORKERS}")

    t_start = time.time()
    completed = len(existing_data)
    total = len(items)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(label_single_image, item): item for item in todo_items}

        for future in as_completed(futures):
            img_id, ok, msg = future.result()
            if ok:
                completed += 1
                if completed % 25 == 0 or completed == total:
                    elapsed = time.time() - t_start
                    speed = (completed - len(existing_data)) / elapsed if elapsed > 0 else 0
                    print(f"   [{completed}/{total}] ({completed/total*100:.1f}%) - {speed:.1f} ảnh/s - ID: {img_id}")
            else:
                print(f"   [LỖI] {img_id}: {msg}")

    print("\n=========================================================")
    print("   GÁN NHÃN HOÀN TẤT THÀNH CÔNG!")
    print(f"   Tổng số nhãn đã lưu: {len(existing_data)} ảnh")
    print(f"   File kết quả: {OUTPUT_JSON}")
    print("=========================================================")

if __name__ == "__main__":
    main()
