# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script tự động gán nhãn 4 góc hóa đơn bằng Gemini 3.5 Flash cho toàn bộ ảnh MC-OCR (~1,534 ảnh)
      sử dụng Native 2D Grounding chống cắt lẹm, đa luồng tốc độ cao, tự động resume.
"""

import os
import sys
import glob
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

# Nạp API key
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv("backend/.env")
        API_KEY = os.environ.get("GEMINI_API_KEY", "")
    except Exception:
        pass

if not API_KEY:
    print("[LỖI] Không tìm thấy GEMINI_API_KEY!")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

OUTPUT_JSON = "mcocr_full_gemini_labels.json"
SEED_JSON = "gemini_mcocr_500_labels.json"
MAX_WORKERS = 8

PROMPT = """Detect the entire physical receipt paper sheet in the image.
Return JSON with:
1. "box_2d": [ymin, xmin, ymax, xmax] (normalized from 0 to 1000) bounding the whole paper sheet from the top edge to the very bottom edge (including barcodes, footer, and borders).
2. "corners": [[y_top_left, x_top_left], [y_top_right, x_top_right], [y_bottom_right, x_bottom_right], [y_bottom_left, x_bottom_left]] (normalized from 0 to 1000) corresponding to the four physical paper corners.
DO NOT crop through the receipt body or cut off the bottom barcode! The bounding box and corners MUST enclose the full physical receipt sheet."""

# Khởi tạo data
full_data = {}

# 1. Kế thừa từ SEED_JSON nếu OUTPUT_JSON chưa có hoặc có ít hơn
if os.path.exists(OUTPUT_JSON):
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        print(f"[RESUME] Đã nạp {len(full_data)} nhãn từ {OUTPUT_JSON}")
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được {OUTPUT_JSON}: {e}")
        full_data = {}

if os.path.exists(SEED_JSON):
    try:
        with open(SEED_JSON, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
        added_from_seed = 0
        for k, v in seed_data.items():
            if k not in full_data and os.path.exists(v.get("image_path", "")):
                full_data[k] = v
                added_from_seed += 1
        if added_from_seed > 0:
            print(f"[SEED] Đã tích hợp {added_from_seed} nhãn sẵn có từ {SEED_JSON}")
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được seed: {e}")

# Thu thập tất cả ảnh MC-OCR
search_patterns = [
    "MASTER_DATASET_ARCHIVE/MC_OCR_Dataset/train_images/train_images/*.jpg",
    "MASTER_DATASET_ARCHIVE/MC_OCR_Dataset/val_images/*.jpg",
    "MASTER_DATASET_ARCHIVE/MC_OCR_Dataset/val_images/*/*.jpg",
]

all_images = []
seen_basenames = set()
for pattern in search_patterns:
    for path in glob.glob(pattern):
        bname = os.path.basename(path)
        if bname not in seen_basenames:
            seen_basenames.add(bname)
            all_images.append(os.path.abspath(path))

print(f"\n[DỮ LIỆU] Tổng số ảnh MC-OCR tìm thấy: {len(all_images)}")

# Lọc ảnh chưa gán nhãn
existing_paths = set(os.path.abspath(v["image_path"]) for v in full_data.values() if "image_path" in v and os.path.exists(v["image_path"]))
pending_images = [p for p in all_images if p not in existing_paths]

print(f"[TIẾN ĐỘ] Đã gán nhãn: {len(existing_paths)} | Cần gán nhãn tiếp: {len(pending_images)}")

lock = threading.Lock()
save_counter = 0

def save_entry(img_id, entry):
    global save_counter
    with lock:
        full_data[img_id] = entry
        save_counter += 1
        if save_counter % 10 == 0 or save_counter == len(pending_images):
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(full_data, f, ensure_ascii=False, indent=2)

def label_image(img_path):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
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

            entry = {
                "image_path": img_path,
                "width": w,
                "height": h,
                "box_2d": [ymin, xmin, ymax, xmax],
                "top_left": top_left,
                "top_right": top_right,
                "bottom_right": bottom_right,
                "bottom_left": bottom_left
            }

            save_entry(img_id, entry)
            return img_id, True, "OK"

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(4 + attempt * 2)
            else:
                time.sleep(1 + attempt)

    return img_id, False, err

def main():
    if not pending_images:
        print("[XONG] Tất cả ảnh đã có nhãn!")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
        return

    print(f"\n[BẮT ĐẦU] Gán nhãn {len(pending_images)} ảnh với {MAX_WORKERS} luồng...")
    t0 = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(label_image, p): p for p in pending_images}
        for future in as_completed(futures):
            p = futures[future]
            try:
                img_id, success, msg = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

            total_done = completed + failed
            if total_done % 25 == 0 or total_done == len(pending_images):
                elapsed = time.time() - t0
                speed = total_done / elapsed if elapsed > 0 else 0
                eta_s = (len(pending_images) - total_done) / speed if speed > 0 else 0
                print(f"[{total_done}/{len(pending_images)}] Thành công: {completed} | Lỗi: {failed} | Tốc độ: {speed:.1f} ảnh/s | Còn lại: {eta_s:.0f}s")

    # Lưu lần cuối
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t0
    print(f"\n[HOÀN TẤT] Tổng số nhãn hiện có: {len(full_data)} | Thời gian: {total_time:.1f}s")

if __name__ == "__main__":
    main()
