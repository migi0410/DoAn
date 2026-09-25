# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script chuyển đổi nhãn MC-OCR từ Gemini 3.5 Flash sang định dạng Ultralytics YOLO-Pose
      (BBox + 4 Keypoints TL-TR-BR-BL), chia Train/Val 85/15 và sinh data.yaml.
"""

import os
import sys
import json
import shutil
import random
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LABELS_JSON = "mcocr_full_gemini_labels.json"
OUTPUT_DIR = "dataset_yolo_real_mcocr"
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

def order_points_clockwise(pts):
    """
    Sắp xếp 4 điểm theo thứ tự chuẩn: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    pts = np.array(pts, dtype=np.float32)
    # Tổng x + y: min là TL, max là BR
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # Hiệu y - x: min là TR, max là BL
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

def main():
    if not os.path.exists(LABELS_JSON):
        print(f"[LỖI] Không tìm thấy file nhãn: {LABELS_JSON}")
        sys.exit(1)

    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"=========================================================")
    print(f"  CHUYỂN ĐỔI DATASET MC-OCR GEMINI SANG YOLO-POSE FORMAT")
    print(f"=========================================================")
    print(f"Tổng số mục nhãn trong file: {len(data)}")

    valid_samples = []
    for img_id, item in data.items():
        img_path = item.get("image_path", "")
        if not os.path.exists(img_path):
            continue

        tl = item.get("top_left")
        tr = item.get("top_right")
        br = item.get("bottom_right")
        bl = item.get("bottom_left")

        if not (tl and tr and br and bl):
            continue

        raw_pts = [tl, tr, br, bl]
        # Kiểm tra tính hợp lệ trong khoảng [0, 1]
        all_valid = True
        for pt in raw_pts:
            if not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                all_valid = False
                break
        if not all_valid:
            continue

        # Sắp xếp 4 góc chuẩn xác TL -> TR -> BR -> BL
        ordered_pts = order_points_clockwise(raw_pts)
        pts_arr = np.array(ordered_pts)

        # Tính BBox bao quanh
        xmin = float(pts_arr[:, 0].min())
        xmax = float(pts_arr[:, 0].max())
        ymin = float(pts_arr[:, 1].min())
        ymax = float(pts_arr[:, 1].max())

        w = xmax - xmin
        h = ymax - ymin
        xc = xmin + w / 2.0
        yc = ymin + h / 2.0

        # Lọc bỏ các bounding box quá nhỏ hoặc dị thường
        if w < 0.05 or h < 0.05:
            continue

        valid_samples.append({
            "img_id": img_id,
            "img_path": img_path,
            "bbox": (xc, yc, w, h),
            "kpts": ordered_pts
        })

    print(f"Số mẫu hợp lệ sau kiểm tra chất lượng: {len(valid_samples)}")

    # Chia train / val
    random.seed(RANDOM_SEED)
    random.shuffle(valid_samples)

    split_idx = int(len(valid_samples) * TRAIN_RATIO)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]

    print(f"Tập Huấn luyện (Train): {len(train_samples)} ảnh ({TRAIN_RATIO*100:.0f}%)")
    print(f"Tập Kiểm thử   (Val):   {len(val_samples)} ảnh ({(1-TRAIN_RATIO)*100:.0f}%)")

    # Chuẩn bị cấu trúc thư mục
    train_img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    val_img_dir = os.path.join(OUTPUT_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "val")
    preview_dir = os.path.join(OUTPUT_DIR, "sample_checks")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, preview_dir]:
        os.makedirs(d, exist_ok=True)

    def write_dataset_split(samples, img_dir, lbl_dir, split_name):
        print(f"\nĐang ghi tập {split_name} ({len(samples)} mẫu)...")
        for idx, s in enumerate(samples):
            bname = os.path.basename(s["img_path"])
            dest_img = os.path.join(img_dir, bname)
            if not os.path.exists(dest_img):
                shutil.copy2(s["img_path"], dest_img)

            # File nhãn YOLO Pose
            lbl_name = os.path.splitext(bname)[0] + ".txt"
            dest_lbl = os.path.join(lbl_dir, lbl_name)

            xc, yc, w, h = s["bbox"]
            kpts = s["kpts"] # [TL, TR, BR, BL]

            # Định dạng: class_id xc yc w h (x y v)*4
            kpt_str = " ".join([f"{p[0]:.6f} {p[1]:.6f} 2" for p in kpts])
            line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {kpt_str}\n"

            with open(dest_lbl, "w", encoding="utf-8") as lf:
                lf.write(line)

            # Vẽ ảnh kiểm tra trực quan cho 5 mẫu đầu tiên
            if idx < 5:
                try:
                    img_cv = cv2.imread(s["img_path"])
                    if img_cv is not None:
                        ih, iw = img_cv.shape[:2]
                        # Vẽ BBox
                        bx1, by1 = int((xc - w/2) * iw), int((yc - h/2) * ih)
                        bx2, by2 = int((xc + w/2) * iw), int((yc + h/2) * ih)
                        cv2.rectangle(img_cv, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                        # Vẽ 4 góc và đa giác
                        pts = [(int(p[0] * iw), int(p[1] * ih)) for p in kpts]
                        cv2.polylines(img_cv, [np.array(pts, np.int32)], True, (0, 255, 255), 3)

                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # TL, TR, BR, BL
                        labels = ["TL", "TR", "BR", "BL"]
                        for pt, c, lbl in zip(pts, colors, labels):
                            cv2.circle(img_cv, pt, 8, c, -1)
                            cv2.putText(img_cv, lbl, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

                        chk_path = os.path.join(preview_dir, f"{split_name}_{idx+1}_{bname}")
                        cv2.imwrite(chk_path, img_cv)
                except Exception:
                    pass

    write_dataset_split(train_samples, train_img_dir, train_lbl_dir, "train")
    write_dataset_split(val_samples, val_img_dir, val_lbl_dir, "val")

    # Tạo data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    abs_output_dir = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    yaml_content = f"""# Cấu hình dataset YOLO11-Pose cho ảnh thực tế MC-OCR
path: {abs_output_dir}
train: images/train
val: images/val

# Cấu hình 4 keypoints góc hóa đơn: TL, TR, BR, BL
kpt_shape: [4, 3] # (x, y, visibility)
flip_idx: [1, 0, 3, 2] # Lật ngang: TL <-> TR, BL <-> BR

names:
  0: invoice
"""
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"\n[HOÀN TẤT] File cấu hình huấn luyện lưu tại: {yaml_path}")
    print(f"Các ảnh kiểm tra trực quan lưu tại: {preview_dir}")
    print("=========================================================")

if __name__ == "__main__":
    main()
