# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script sinh 3000 ảnh hóa đơn giả lập kèm nhãn YOLO Pose (4 góc: TL, TR, BR, BL)
      phục vụ bài toán Preprocessing / Document Corner Detection & Dewarping / Border Cropping.
"""

import os
import sys
import json
import random
import time
import argparse
import cv2
import numpy as np
from playwright.sync_api import sync_playwright

# Đảm bảo in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from invoice_templates import TEMPLATES_MAP
from synthetic_generator import generate_random_invoice_data, render_html_with_data
from receipt_augmenter import ReceiptAugmenter


class BackgroundGenerator:
    """Sinh đa dạng nền chụp ảnh toán học (wood, marble, desk, concrete, granite, tile, plain)"""
    @staticmethod
    def generate(w, h, bg_type=None):
        if bg_type is None:
            bg_type = random.choice(["wood", "desk", "concrete", "marble", "granite", "tile", "plain"])
            
        if bg_type == "wood":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            # Màu gỗ ấm ngẫu nhiên
            base_b = random.randint(30, 60)
            base_g = random.randint(65, 100)
            base_r = random.randint(110, 150)
            bg[:, :] = [base_b, base_g, base_r]
            noise = np.random.normal(0, 18, (h, w, 1))
            noise = cv2.GaussianBlur(noise, (1, 99), 0)
            if len(noise.shape) == 2:
                noise = noise[:, :, np.newaxis]
            bg = np.clip(bg.astype(np.int16) + (noise * 0.6).astype(np.int16), 0, 255).astype(np.uint8)
            y_idx, x_idx = np.indices((h, w))
            ring = np.sin(x_idx / 45.0 + np.sin(y_idx / 110.0) * 2.0) * 12
            bg = np.clip(bg.astype(np.int16) + ring[:, :, np.newaxis], 0, 255).astype(np.uint8)
            return bg

        elif bg_type == "desk":
            # Bàn làm việc tối màu + vignette
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            tone = random.randint(35, 70)
            bg[:, :] = [tone, tone - 5, tone - 10]
            y_idx, x_idx = np.indices((h, w))
            cx, cy = w // 2, h // 2
            dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            vignette = 1.0 - (dist / max_dist) * random.uniform(0.3, 0.5)
            bg = (bg.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
            noise = np.random.normal(0, 4, (h, w, 3))
            return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif bg_type == "concrete":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            gray = random.randint(150, 210)
            bg[:, :] = [gray, gray, gray]
            noise = np.random.normal(0, 12, (h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(bg, (3, 3), 0)

        elif bg_type == "marble":
            # Mặt đá cẩm thạch trắng sáng có vân xám mờ
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = [random.randint(220, 240)] * 3
            y_idx, x_idx = np.indices((h, w))
            vein = np.sin(x_idx / 30.0 + np.cos(y_idx / 40.0) * 3.0) * 15
            vein += np.cos(x_idx / 60.0 - y_idx / 50.0) * 10
            noise = np.random.normal(0, 5, (h, w))
            vein_full = vein + noise
            bg = np.clip(bg.astype(np.int16) - np.abs(vein_full[:, :, np.newaxis] * 1.5).astype(np.int16), 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(bg, (3, 3), 0)

        elif bg_type == "granite":
            # Đá hoa cương chấm hạt đen trắng xám
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            base_gray = random.randint(60, 120)
            bg[:, :] = [base_gray, base_gray, base_gray]
            speckles = np.random.normal(0, 25, (h, w, 3))
            bg = np.clip(bg.astype(np.float32) + speckles, 0, 255).astype(np.uint8)
            return cv2.GaussianBlur(bg, (3, 3), 0)

        elif bg_type == "tile":
            # Gạch lát nền kèm đường ron gạch (grout lines)
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = [random.randint(180, 220), random.randint(180, 220), random.randint(190, 230)]
            tile_size = random.randint(150, 250)
            for x in range(0, w, tile_size):
                cv2.line(bg, (x, 0), (x, h), (140, 140, 140), 2)
            for y in range(0, h, tile_size):
                cv2.line(bg, (0, y), (w, y), (140, 140, 140), 2)
            noise = np.random.normal(0, 6, (h, w, 3))
            return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        else: # plain
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            b = random.randint(100, 210)
            g = random.randint(100, 210)
            r = random.randint(100, 210)
            bg[:, :] = [b, g, r]
            noise = np.random.normal(0, 5, (h, w, 3))
            return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def transform_receipt_onto_canvas(clean_receipt_img, canvas_size=640):
    """
    Đặt hóa đơn lên canvas nền với góc xoay ngẫu nhiên, phối cảnh (keystone), 
    khoảng đệm rìa (margin), nếp nhăn và bóng đổ.
    Trả về:
      final_img (canvas_size, canvas_size, 3)
      dst_pts: numpy array shape (4, 2) theo thứ tự chuẩn [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    h_orig, w_orig = clean_receipt_img.shape[:2]
    
    # 1. Thêm nếp nhăn & nếp gấp ngẫu nhiên
    if random.random() < 0.6:
        clean_receipt_img, _ = ReceiptAugmenter.apply_folds_and_creases(
            clean_receipt_img, [], 
            num_folds=random.randint(1, 3), 
            fold_amplitude=random.uniform(1.0, 3.5)
        )
    
    # 2. Phai màu mực & vệt đầu in nhiệt ngẫu nhiên
    if random.random() < 0.4:
        clean_receipt_img = ReceiptAugmenter.apply_thermal_fading_and_streaks(
            clean_receipt_img, 
            fade_intensity=random.uniform(0.05, 0.25), 
            streak_probability=random.uniform(0.1, 0.3)
        )

    # 3. Tính tỉ lệ scale sao cho hóa đơn nằm vừa trong canvas và có rìa (margin 10% - 30%)
    target_coverage = random.uniform(0.55, 0.88)
    scale = min((canvas_size * target_coverage) / w_orig, (canvas_size * target_coverage) / h_orig)
    new_w = max(50, int(w_orig * scale))
    new_h = max(80, int(h_orig * scale))
    
    resized = cv2.resize(clean_receipt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Tọa độ gốc 4 góc tâm (0, 0): [TL, TR, BR, BL]
    half_w, half_h = new_w / 2.0, new_h / 2.0
    pts_centered = np.float32([
        [-half_w, -half_h],  # TL
        [ half_w, -half_h],  # TR
        [ half_w,  half_h],  # BR
        [-half_w,  half_h]   # BL
    ])

    # 5. Xoay ngẫu nhiên góc theta (-30 đến +30 độ)
    angle_deg = random.uniform(-30, 30)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    pts_rotated = np.dot(pts_centered, R.T)

    # 6. Dịch chuyển tâm quanh giữa canvas
    max_shift = canvas_size * 0.08
    cx = canvas_size / 2.0 + random.uniform(-max_shift, max_shift)
    cy = canvas_size / 2.0 + random.uniform(-max_shift, max_shift)
    pts_canvas = pts_rotated + np.array([cx, cy])

    # 7. Thêm biến dạng phối cảnh (Keystone / Tilt distortion)
    jitter_amount = min(new_w, new_h) * random.uniform(0.02, 0.08)
    for i in range(4):
        pts_canvas[i] += [random.uniform(-jitter_amount, jitter_amount), random.uniform(-jitter_amount, jitter_amount)]

    # Kẹp tọa độ bên trong canvas (đảm bảo không bị bay ra ngoài canvas)
    pad = 5.0
    pts_canvas[:, 0] = np.clip(pts_canvas[:, 0], pad, canvas_size - pad)
    pts_canvas[:, 1] = np.clip(pts_canvas[:, 1], pad, canvas_size - pad)

    # 8. Ma trận biến đổi phối cảnh M
    src_corners = np.float32([
        [0, 0],
        [new_w - 1, 0],
        [new_w - 1, new_h - 1],
        [0, new_h - 1]
    ])
    dst_corners = np.float32(pts_canvas)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # 9. Warp ảnh hóa đơn & mask
    warped_receipt = cv2.warpPerspective(
        resized, M, (canvas_size, canvas_size), 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask, M, (canvas_size, canvas_size), 
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 10. Tạo nền ngẫu nhiên
    bg = BackgroundGenerator.generate(canvas_size, canvas_size)

    # 11. Bóng đổ Drop Shadow mềm
    shadow_offset_x = random.randint(4, 12)
    shadow_offset_y = random.randint(4, 12)
    blur_k = random.choice([15, 21, 29])
    shadow_mask = cv2.GaussianBlur(warped_mask, (blur_k, blur_k), 0)
    M_shift = np.float32([[1, 0, shadow_offset_x], [0, 1, shadow_offset_y]])
    shadow_shifted = cv2.warpAffine(shadow_mask, M_shift, (canvas_size, canvas_size))
    shadow_factor = 1.0 - (shadow_shifted / 255.0) * random.uniform(0.35, 0.55)
    bg = (bg.astype(np.float32) * shadow_factor[:, :, np.newaxis]).astype(np.uint8)

    # 12. Trộn ảnh hóa đơn lên nền
    mask_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
    final_img = np.where(mask_3c == 255, warped_receipt, bg)

    # 13. Thêm đèn flash hoặc bóng đổ chéo ngẫu nhiên
    if random.random() < 0.5:
        final_img = ReceiptAugmenter.apply_shadows_and_flashlight(
            final_img, 
            shadow_intensity=random.uniform(0.1, 0.35), 
            flash_intensity=random.uniform(0.1, 0.3)
        )

    # 14. Nhiễu hạt hoặc mờ nhẹ (mô phỏng camera thực tế)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        final_img = cv2.GaussianBlur(final_img, (k, k), 0)

    return final_img, dst_corners


def convert_corners_to_yolo_pose(corners, canvas_size=640):
    """
    Format YOLO Pose:
    class_id bbox_x bbox_y bbox_w bbox_h kpt1_x kpt1_y 2 kpt2_x kpt2_y 2 kpt3_x kpt3_y 2 kpt4_x kpt4_y 2
    """
    norm_pts = []
    for x, y in corners:
        nx = max(0.0, min(1.0, float(x) / canvas_size))
        ny = max(0.0, min(1.0, float(y) / canvas_size))
        norm_pts.append((nx, ny))

    xs = [p[0] for p in norm_pts]
    ys = [p[1] for p in norm_pts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bbox_x = (min_x + max_x) / 2.0
    bbox_y = (min_y + max_y) / 2.0
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    kpts_str = " ".join([f"{p[0]:.6f} {p[1]:.6f} 2" for p in norm_pts])
    line = f"0 {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f} {kpts_str}\n"
    return line


def main():
    parser = argparse.ArgumentParser(description="Sinh tập dữ liệu 3000 ảnh YOLO Pose cắt hóa đơn")
    parser.add_argument("--total_count", type=int, default=3000, help="Tổng số ảnh cần sinh (mặc định 3000)")
    parser.add_argument("--num_base", type=int, default=150, help="Số lượng mẫu hóa đơn gốc sạch Playwright render (mặc định 150)")
    parser.add_argument("--output_dir", default="dataset_yolo_crop", help="Thư mục xuất dataset YOLO")
    parser.add_argument("--img_size", type=int, default=640, help="Kích thước ảnh vuông YOLO (mặc định 640x640)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    print("=========================================================")
    print("   SINH DỮ LIỆU TỰ ĐỘNG CHO YOLO POSE (CẮT HÓA ĐƠN)    ")
    print("=========================================================")
    print(f"Tổng số ảnh mục tiêu: {args.total_count}")
    print(f"Kích thước ảnh YOLO: {args.img_size}x{args.img_size}")
    print(f"Thư mục lưu trữ: {output_dir}")
    print(f"Số hóa đơn gốc Playwright: {args.num_base}")

    # Tạo thư mục theo cấu trúc YOLO
    dirs = {
        "train_img": os.path.join(output_dir, "images", "train"),
        "val_img": os.path.join(output_dir, "images", "val"),
        "train_lbl": os.path.join(output_dir, "labels", "train"),
        "val_lbl": os.path.join(output_dir, "labels", "val"),
        "debug": os.path.join(output_dir, "debug_samples")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    templates_list = list(TEMPLATES_MAP.keys())
    print(f"\n[1/3] Đang render {args.num_base} mẫu hóa đơn sạch với {len(templates_list)} templates...")
    clean_invoices_pool = []
    
    t_start = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Tạo contexts
        thermal_ctx = browser.new_context(viewport={"width": 380, "height": 800})
        a4_ctx = browser.new_context(viewport={"width": 794, "height": 1123})
        c45_ctx = browser.new_context(viewport={"width": 650, "height": 450})
        
        thermal_page = thermal_ctx.new_page()
        a4_page = a4_ctx.new_page()
        c45_page = c45_ctx.new_page()

        temp_dir = os.path.join(output_dir, ".temp_clean")
        os.makedirs(temp_dir, exist_ok=True)

        for i in range(args.num_base):
            tmpl = templates_list[i % len(templates_list)]
            data = generate_random_invoice_data(tmpl)
            html = render_html_with_data(data, tmpl)

            if tmpl.startswith("einvoice_"):
                pg = a4_page
            elif tmpl == "receipt_c45_bb":
                pg = c45_page
            else:
                pg = thermal_page

            pg.set_content(html)
            # Chờ nhẹ font chữ
            pg.wait_for_timeout(60)

            temp_img_path = os.path.join(temp_dir, f"clean_{i:04d}.png")
            pg.screenshot(path=temp_img_path, full_page=True)
            
            img_mat = cv2.imread(temp_img_path)
            if img_mat is not None:
                clean_invoices_pool.append(img_mat)

            if (i + 1) % 25 == 0 or (i + 1) == args.num_base:
                print(f"   Đã render {i + 1}/{args.num_base} mẫu sạch ({time.time() - t_start:.1f}s)")

        browser.close()

    print(f"-> Thu thập thành công {len(clean_invoices_pool)} mẫu hóa đơn sạch đa dạng!")

    # Bước 2: Sinh 3000 ảnh tăng cường
    num_train = int(args.total_count * 0.8)
    num_val = args.total_count - num_train
    print(f"\n[2/3] Bắt đầu sinh và tăng cường {args.total_count} ảnh (Train: {num_train}, Val: {num_val})...")

    splits = [("train", num_train), ("val", num_val)]
    debug_samples_count = 0
    t_gen_start = time.time()
    total_idx = 0

    for split_name, count in splits:
        img_dest = dirs[f"{split_name}_img"]
        lbl_dest = dirs[f"{split_name}_lbl"]

        for j in range(count):
            total_idx += 1
            sample_name = f"invoice_{split_name}_{j+1:05d}"
            
            # Chọn ngẫu nhiên 1 mẫu sạch từ pool
            base_img = random.choice(clean_invoices_pool)
            
            # Biến đổi lên canvas với góc xoay, nền mặt bàn, nếp nhăn, bóng đổ
            final_img, dst_corners = transform_receipt_onto_canvas(base_img, canvas_size=args.img_size)

            # Lưu ảnh JPG (chất lượng 92 để nhẹ và nhanh)
            out_img_path = os.path.join(img_dest, f"{sample_name}.jpg")
            cv2.imwrite(out_img_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Chuyển đổi và lưu nhãn YOLO Pose TXT
            out_lbl_path = os.path.join(lbl_dest, f"{sample_name}.txt")
            yolo_line = convert_corners_to_yolo_pose(dst_corners, canvas_size=args.img_size)
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write(yolo_line)

            # Lưu một số mẫu trực quan vào debug_samples
            debug_interval = max(1, count // 5)
            if debug_samples_count < 10 and j % debug_interval == 0:
                debug_samples_count += 1
                debug_canvas = final_img.copy()
                pts_int = np.int32(dst_corners)
                # Vẽ đa giác nối 4 góc (màu vàng)
                cv2.polylines(debug_canvas, [pts_int], True, (0, 255, 255), 2)
                # Đánh dấu 4 góc: TL (Đỏ), TR (Xanh lá), BR (Xanh dương), BL (Vàng)
                corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                corner_names = ["TL", "TR", "BR", "BL"]
                for k, (pt, col, name) in enumerate(zip(pts_int, corner_colors, corner_names)):
                    cv2.circle(debug_canvas, tuple(pt), 7, col, -1)
                    cv2.putText(debug_canvas, name, (pt[0] + 8, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                debug_path = os.path.join(dirs["debug"], f"debug_{split_name}_{j+1:04d}.jpg")
                cv2.imwrite(debug_path, debug_canvas)

            if total_idx % 500 == 0 or total_idx == args.total_count:
                elapsed = time.time() - t_gen_start
                fps = total_idx / elapsed if elapsed > 0 else 0
                print(f"   Đã sinh: {total_idx}/{args.total_count} ảnh [{split_name}] ({fps:.1f} ảnh/giây)...")

    # Bước 3: Tạo file data.yaml
    print("\n[3/3] Đang tạo file cấu hình data.yaml...")
    yaml_content = f"""# Cấu hình Dataset YOLOv8 / YOLO11 Pose cho bài toán Cắt Hóa Đơn
path: {output_dir.replace(chr(92), '/')}
train: images/train
val: images/val

# 4 góc hóa đơn: TL (0), TR (1), BR (2), BL (3)
kpt_shape: [4, 2]

# Nếu lật ảnh ngang (horizontal flip): TL <-> TR (0 <-> 1), BL <-> BR (3 <-> 2)
flip_idx: [1, 0, 3, 2]

names:
  0: invoice
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Dọn dẹp thư mục tạm
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - t_start
    print("\n=========================================================")
    print(f"   HOÀN THÀNH XUẤT SẮC TRONG {total_time:.1f} GIÂY ({total_time/60:.2f} phút)!")
    print(f"   - Train images: {num_train} ảnh tại images/train/")
    print(f"   - Val images:   {num_val} ảnh tại images/val/")
    print(f"   - File data.yaml: {yaml_path}")
    print(f"   - Ảnh kiểm tra nhãn: {dirs['debug']}")
    print("=========================================================")


if __name__ == "__main__":
    main()
