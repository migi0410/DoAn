# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script suy luận (Inference): Nhận diện 4 góc bằng YOLO Pose,
      cắt bỏ rìa nền và nắn thẳng (Dewarp / Perspective Transform) ảnh hóa đơn.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    Dựa trên quy luật hình học:
    - TL: x + y nhỏ nhất
    - BR: x + y lớn nhất
    - TR: x - y lớn nhất (hoặc y - x nhỏ nhất)
    - BL: y - x lớn nhất (hoặc x - y nhỏ nhất)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right

    diff = np.diff(pts, axis=1) # y - x
    rect[1] = pts[np.argmin(diff)] # Top-Right (x lớn, y nhỏ => y - x âm nhất)
    rect[3] = pts[np.argmax(diff)] # Bottom-Left (x nhỏ, y lớn => y - x dương nhất)
    return rect

def crop_and_dewarp(image, keypoints):
    """
    Thực hiện phép biến đổi phối cảnh (Perspective Transform) từ 4 điểm keypoints.
    """
    # Keypoints shape: (4, 2)
    # Có thể dùng trực tiếp thứ tự keypoints được train hoặc order_points để an toàn
    pts_src = np.float32(keypoints)

    # 1. Tính chiều rộng tối đa của hóa đơn mới
    width_top = np.linalg.norm(pts_src[1] - pts_src[0])
    width_bottom = np.linalg.norm(pts_src[2] - pts_src[3])
    max_w = int(max(width_top, width_bottom))

    # 2. Tính chiều cao tối đa của hóa đơn mới
    height_left = np.linalg.norm(pts_src[3] - pts_src[0])
    height_right = np.linalg.norm(pts_src[2] - pts_src[1])
    max_h = int(max(height_left, height_right))

    if max_w <= 0 or max_h <= 0:
        return None

    # 3. Tọa độ đích hình chữ nhật phẳng
    pts_dst = np.float32([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ])

    # 4. Ma trận phối cảnh và làm phẳng
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
    return warped

def fix_missing_corners(kpts, bbox=None):
    pts = np.copy(kpts)
    zeros = [i for i in range(4) if np.all(pts[i] == 0) or (pts[i][0] == 0 and pts[i][1] == 0)]
    if len(zeros) == 1:
        z = zeros[0]
        if z == 3: # BL thiếu
            pts[3] = pts[0] + (pts[2] - pts[1])
        elif z == 2: # BR thiếu
            pts[2] = pts[1] + (pts[3] - pts[0])
        elif z == 1: # TR thiếu
            pts[1] = pts[0] + (pts[2] - pts[3])
        elif z == 0: # TL thiếu
            pts[0] = pts[1] + (pts[3] - pts[2])
    elif len(zeros) > 1 and bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        pts = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ], dtype=np.float32)
    return pts

def process_single_image(model, image_path, output_dir, save_debug=True, conf_thresh=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[CẢNH BÁO] Không đọc được ảnh: {image_path}")
        return False

    results = model(image_path, conf=conf_thresh, verbose=False)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"[BỎ QUA] Không tìm thấy hóa đơn trong ảnh: {os.path.basename(image_path)}")
        return False

    # Lấy keypoints và bounding box
    bbox_px = results[0].boxes.xyxy.cpu().numpy()[0]
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        raw_kpts = results[0].keypoints.xy.cpu().numpy()[0]
        kpts = fix_missing_corners(raw_kpts, bbox=bbox_px)
    else:
        xmin, ymin, xmax, ymax = bbox_px
        kpts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)

    # Cắt và nắn phẳng
    cropped = crop_and_dewarp(img, kpts)
    if cropped is None:
        return False

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_crop_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
    cv2.imwrite(out_crop_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Lưu ảnh debug nếu yêu cầu
    if save_debug:
        debug_dir = os.path.join(output_dir, "debug_detection")
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = img.copy()
        pts_int = np.int32(kpts)
        cv2.polylines(debug_img, [pts_int], True, (0, 255, 255), 3)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # TL, TR, BR, BL
        names = ["TL", "TR", "BR", "BL"]
        for pt, col, nm in zip(pts_int, colors, names):
            cv2.circle(debug_img, tuple(pt), 8, col, -1)
            cv2.putText(debug_img, nm, (pt[0] + 10, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        debug_path = os.path.join(debug_dir, f"{base_name}_detected.jpg")
        cv2.imwrite(debug_path, debug_img)

    return True

def main():
    parser = argparse.ArgumentParser(description="Cắt rìa và nắn phẳng hóa đơn bằng YOLO Pose")
    parser.add_argument("--weights", default="runs/pose/invoice_crop/weights/best.pt", help="File trọng số best.pt")
    parser.add_argument("--source", required=True, help="Đường dẫn file ảnh đơn lẻ hoặc thư mục ảnh cần xử lý")
    parser.add_argument("--output_dir", default="cropped_invoices", help="Thư mục lưu ảnh đã cắt")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy (Confidence threshold)")
    parser.add_argument("--no_debug", action="store_true", help="Không lưu ảnh vẽ 4 góc để debug")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[LỖI] Không tìm thấy file trọng số tại: {args.weights}")
        print("Vui lòng huấn luyện model trước bằng lệnh: python train_yolo_crop.py")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=========================================================")
    print("      TIỀN XỬ LÝ: CẮT RÌA & NẮN PHẲNG HÓA ĐƠN            ")
    print("=========================================================")
    print(f"Model:      {args.weights}")
    print(f"Nguồn ảnh:  {args.source}")
    print(f"Đầu ra:     {args.output_dir}")
    print("=========================================================\n")

    model = YOLO(args.weights)

    if os.path.isfile(args.source):
        image_files = [args.source]
    elif os.path.isdir(args.source):
        exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        image_files = [
            os.path.join(args.source, f) for f in os.listdir(args.source)
            if os.path.splitext(f)[1].lower() in exts
        ]
    else:
        print(f"[LỖI] Đường dẫn nguồn không hợp lệ: {args.source}")
        sys.exit(1)

    print(f"Đang xử lý {len(image_files)} ảnh...")
    success_count = 0
    for f in image_files:
        if process_single_image(model, f, args.output_dir, save_debug=not args.no_debug, conf_thresh=args.conf):
            success_count += 1

    print(f"\n[HOÀN TẤT] Đã cắt thành công {success_count}/{len(image_files)} ảnh vào thư mục: {args.output_dir}")

if __name__ == "__main__":
    main()
