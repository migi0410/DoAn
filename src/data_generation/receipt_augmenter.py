# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Module tăng cường dữ liệu ảnh hóa đơn giả lập (Receipt Augmentation)
      Hỗ trợ mô phỏng nếp gấp, bóng mờ, ánh sáng, chụp nghiêng, lồng nền mặt bàn và bóng đổ drop shadow.
"""

import cv2
import numpy as np
import random
import math

class ReceiptAugmenter:
    @staticmethod
    def generate_procedural_background(w, h, bg_type="wood"):
        """
        Sinh nền mặt bàn chụp ảnh toán học (procedural background) bằng numpy/opencv.
        """
        if bg_type == "wood":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            # Màu BGR cơ bản của gỗ/vàng ấm
            bg[:, :] = [45, 82, 125] 
            # Tạo nhiễu hạt dọc
            noise = np.random.normal(0, 15, (h, w, 1))
            # Blur mạnh theo chiều dọc để tạo vân gỗ thẳng
            noise = cv2.GaussianBlur(noise, (1, 99), 0)
            if len(noise.shape) == 2:
                noise = noise[:, :, np.newaxis]
            grain = (noise * 0.5).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)
            # Thêm các vân vòng cung nhẹ
            y_indices, x_indices = np.indices((h, w))
            ring = np.sin(x_indices / 40.0 + np.sin(y_indices / 100.0) * 2.0) * 10
            bg = np.clip(bg.astype(np.int16) + ring[:, :, np.newaxis], 0, 255).astype(np.uint8)
            return bg
            
        elif bg_type == "concrete":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            # Màu xám xi măng cơ bản
            bg[:, :] = [180, 180, 180]
            # Tạo nhiễu hạt đều
            noise = np.random.normal(0, 10, (h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            bg = cv2.GaussianBlur(bg, (3, 3), 0)
            return bg
            
        elif bg_type == "desk":
            # Mặt bàn tối màu có hiệu ứng vignette
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = [60, 50, 45]
            y_indices, x_indices = np.indices((h, w))
            cx, cy = w // 2, h // 2
            dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            vignette = 1.0 - (dist / max_dist) * 0.4
            bg = (bg.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)
            # Thêm nhiễu mịn
            noise = np.random.normal(0, 3, (h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return bg
            
        else:
            # Nền trắng mặc định
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = [255, 255, 255]
            return bg

    @staticmethod
    def apply_perspective_warp(image, annotations, max_distortion=0.06, bg_type="white"):
        """
        Mô phỏng chụp ảnh nghiêng phối cảnh và hòa trộn ảnh hóa đơn lên nền mặt bàn kèm drop shadow.
        """
        h, w = image.shape[:2]
        
        # Điểm góc ban đầu
        src_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])
        
        # Độ lệch tối đa dựa trên kích thước ảnh
        dx_max = max_distortion * w
        dy_max = max_distortion * h
        
        # Điểm góc đích bị lệch ngẫu nhiên
        dst_pts = np.float32([
            [random.uniform(0, dx_max), random.uniform(0, dy_max)],
            [w - 1 - random.uniform(0, dx_max), random.uniform(0, dy_max)],
            [w - 1 - random.uniform(0, dx_max), h - 1 - random.uniform(0, dy_max)],
            [random.uniform(0, dx_max), h - 1 - random.uniform(0, dy_max)]
        ])
        
        # Tính ma trận phối cảnh
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp ảnh hóa đơn và mask hóa đơn (nền trong suốt/đen)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_img = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Sinh nền mặt bàn tương ứng
        bg = ReceiptAugmenter.generate_procedural_background(w, h, bg_type)
        
        # Tạo bóng đổ Drop Shadow mềm mịn dịch chuyển
        if bg_type != "white":
            shadow_offset_x = random.randint(5, 12)
            shadow_offset_y = random.randint(5, 12)
            shadow_mask = warped_mask.copy()
            # Làm mờ mịn
            blur_k = int(max(w, h) * 0.03) | 1
            shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_k, blur_k), 0)
            # Dịch chuyển bóng đổ theo offset
            M_shift = np.float32([[1, 0, shadow_offset_x], [0, 1, shadow_offset_y]])
            shadow_mask_shifted = cv2.warpAffine(shadow_mask, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Làm tối nền ở vùng có bóng đổ (bóng mềm mại nhân với 0.4)
            shadow_factor = 1.0 - (shadow_mask_shifted / 255.0) * random.uniform(0.3, 0.45)
            bg = (bg.astype(np.float32) * shadow_factor[:, :, np.newaxis]).astype(np.uint8)
            
        # Trộn ảnh hóa đơn và nền
        warped_mask_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
        blended_img = np.where(warped_mask_3c == 255, warped_img, bg)
        
        # Cập nhật tọa độ các bounding box
        updated_annotations = []
        for ann in annotations:
            box = ann["box"] # [xmin, ymin, xmax, ymax]
            
            # Lấy 4 góc của bounding box ban đầu
            box_pts = np.array([
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]]
            ], dtype=np.float32)
            
            # Chuyển đổi tọa độ phối cảnh của 4 góc
            box_pts_hom = np.hstack([box_pts, np.ones((4, 1), dtype=np.float32)])
            transformed_pts = M.dot(box_pts_hom.T).T
            transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]
            
            # Tính lại bounding box bao ngoài (axis-aligned bounding box)
            x_coords = transformed_pts[:, 0]
            y_coords = transformed_pts[:, 1]
            
            xmin_new = max(0, int(np.min(x_coords)))
            ymin_new = max(0, int(np.min(y_coords)))
            xmax_new = min(w - 1, int(np.max(x_coords)))
            ymax_new = min(h - 1, int(np.max(y_coords)))
            
            # Bỏ qua nếu box quá nhỏ hoặc bay khỏi khung hình
            if (xmax_new - xmin_new) > 2 and (ymax_new - ymin_new) > 2:
                new_ann = ann.copy()
                new_ann["box"] = [xmin_new, ymin_new, xmax_new, ymax_new]
                updated_annotations.append(new_ann)
                
        return blended_img, updated_annotations

    @staticmethod
    def apply_shadows_and_flashlight(image, shadow_intensity=0.3, flash_intensity=0.3):
        """
        Mô phỏng ánh sáng loang lổ và hiệu ứng flash.
        """
        h, w = image.shape[:2]
        augmented = image.copy().astype(np.float32)
        
        # 1. Tạo bóng mờ ngẫu nhiên (Shadow)
        if shadow_intensity > 0:
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            num_pts = random.randint(3, 5)
            pts = []
            border_pts = [
                (0, random.randint(0, h)),
                (w - 1, random.randint(0, h)),
                (random.randint(0, w), 0),
                (random.randint(0, w), h - 1)
            ]
            for _ in range(num_pts):
                pts.append(random.choice(border_pts))
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(shadow_mask, [pts], 1.0)
            
            blur_ksize = int(max(w, h) * 0.15) | 1
            shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_ksize, blur_ksize), 0)
            
            shadow_factor = 1.0 - (shadow_mask * shadow_intensity)
            for c in range(3):
                augmented[:, :, c] *= shadow_factor
                
        # 2. Tạo đèn flash ngẫu nhiên (Flashlight)
        if flash_intensity > 0:
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            max_r = max(w, h) * random.uniform(0.4, 0.8)
            
            y_indices, x_indices = np.indices((h, w))
            dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
            
            flash_mask = np.maximum(0.0, 1.0 - np.sqrt(dist_sq) / max_r)
            flash_mask = 0.5 * (1.0 + np.cos(np.pi * (1.0 - flash_mask))) * flash_mask
            
            flash_factor = flash_mask * flash_intensity * 255.0
            for c in range(3):
                augmented[:, :, c] += flash_factor
                
        return np.clip(augmented, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_folds_and_creases(image, annotations, num_folds=2, fold_amplitude=3):
        """
        Mô phỏng nếp nhăn nếp gấp giấy dạng 3D.
        """
        h, w = image.shape[:2]
        augmented_img = image.copy()
        
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        fold_locations = []
        for _ in range(num_folds):
            is_horizontal = random.choice([True, False])
            if is_horizontal:
                y_fold = random.randint(int(h * 0.15), int(h * 0.85))
                fold_locations.append((True, y_fold))
                sigma = h * 0.08
                amp = random.uniform(-fold_amplitude, fold_amplitude)
                freq = random.uniform(1.0, 2.0)
                displacement = amp * np.sin(2 * np.pi * freq * map_x / w) * np.exp(-np.abs(map_y - y_fold) / sigma)
                map_y += displacement
            else:
                x_fold = random.randint(int(w * 0.15), int(w * 0.85))
                fold_locations.append((False, x_fold))
                sigma = w * 0.08
                amp = random.uniform(-fold_amplitude, fold_amplitude)
                freq = random.uniform(1.0, 2.0)
                displacement = amp * np.sin(2 * np.pi * freq * map_y / h) * np.exp(-np.abs(map_x - x_fold) / sigma)
                map_x += displacement
                
        augmented_img = cv2.remap(augmented_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        # Cập nhật tọa độ annotations
        updated_annotations = []
        for ann in annotations:
            box = ann["box"]
            box_pts = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            new_pts = []
            for pt in box_pts:
                px, py = pt[0], pt[1]
                new_px, new_py = px, py
                for is_horiz, fold_pos in fold_locations:
                    if is_horiz:
                        sigma = h * 0.08
                        amp = random.uniform(-fold_amplitude, fold_amplitude)
                        freq = random.uniform(1.0, 2.0)
                        disp = amp * math.sin(2 * math.pi * freq * px / w) * math.exp(-abs(py - fold_pos) / sigma)
                        new_py += disp
                    else:
                        sigma = w * 0.08
                        amp = random.uniform(-fold_amplitude, fold_amplitude)
                        freq = random.uniform(1.0, 2.0)
                        disp = amp * math.sin(2 * math.pi * freq * py / h) * math.exp(-abs(px - fold_pos) / sigma)
                        new_px += disp
                new_pts.append([new_px, new_py])
                
            new_pts = np.array(new_pts)
            xmin_new = max(0, int(np.min(new_pts[:, 0])))
            ymin_new = max(0, int(np.min(new_pts[:, 1])))
            xmax_new = min(w - 1, int(np.max(new_pts[:, 0])))
            ymax_new = min(h - 1, int(np.max(new_pts[:, 1])))
            
            if (xmax_new - xmin_new) > 2 and (ymax_new - ymin_new) > 2:
                new_ann = ann.copy()
                new_ann["box"] = [xmin_new, ymin_new, xmax_new, ymax_new]
                updated_annotations.append(new_ann)
            else:
                updated_annotations.append(ann)
                
        # Shading đường nếp gấp 3D
        overlay = augmented_img.copy().astype(np.float32)
        for is_horiz, fold_pos in fold_locations:
            if is_horiz:
                cv2.line(overlay, (0, fold_pos - 1), (w, fold_pos - 1), (190, 190, 190), 2)
                cv2.line(overlay, (0, fold_pos + 1), (w, fold_pos + 1), (255, 255, 255), 2)
            else:
                cv2.line(overlay, (fold_pos - 1, 0), (fold_pos - 1, h), (190, 190, 190), 2)
                cv2.line(overlay, (fold_pos + 1, 0), (fold_pos + 1, h), (255, 255, 255), 2)
                
        blur_size = int(max(w, h) * 0.008) | 1
        overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        augmented_img = cv2.addWeighted(augmented_img, 0.75, overlay.astype(np.uint8), 0.25, 0)
        
        return augmented_img, updated_annotations

    @staticmethod
    def apply_thermal_fading_and_streaks(image, fade_intensity=0.3, streak_probability=0.3):
        """
        Mô phỏng lỗi máy in nhiệt: vệt trắng dọc và vùng phai màu.
        """
        h, w = image.shape[:2]
        augmented = image.copy().astype(np.float32)
        
        # 1. Vệt trắng dọc
        if streak_probability > 0:
            num_streaks = random.randint(1, int(w * 0.02 * streak_probability + 2))
            for _ in range(num_streaks):
                x_pos = random.randint(0, w - 1)
                streak_w = random.randint(1, 3)
                opacity = random.uniform(0.4, 0.9)
                streak = np.ones((h, streak_w, 3), dtype=np.float32) * 255.0
                x_end = min(w, x_pos + streak_w)
                actual_w = x_end - x_pos
                augmented[:, x_pos:x_end, :] = (augmented[:, x_pos:x_end, :] * (1.0 - opacity) + streak[:, :actual_w, :] * opacity)
                
        # 2. Phai màu từng vùng
        if fade_intensity > 0:
            num_blobs = random.randint(1, 3)
            for _ in range(num_blobs):
                cx = random.randint(0, w - 1)
                cy = random.randint(0, h - 1)
                rx = random.randint(int(w * 0.1), int(w * 0.35))
                ry = random.randint(int(h * 0.1), int(h * 0.35))
                
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(mask, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, 1.0, -1)
                
                blur_k = int(max(rx, ry) * 1.5) | 1
                mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
                
                fade_factor = mask[:, :, np.newaxis] * fade_intensity
                augmented = augmented * (1.0 - fade_factor) + 255.0 * fade_factor
                
        return np.clip(augmented, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_pipeline(image_path, annotations, perspective_level=0.03, shadow_level=0.25, flash_level=0.2, fold_level=2.0,
                       fade_level=0.0, streak_level=0.0, bg_type="white"):
        """
        Chạy quy trình tăng cường ảnh tích hợp lồng nền mặt bàn và drop shadow.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh tại: {image_path}")
            
        current_ann = [ann.copy() for ann in annotations]
        
        # 1. Nếp gấp & Nếp nhăn giấy
        if fold_level > 0:
            img, current_ann = ReceiptAugmenter.apply_folds_and_creases(
                img, current_ann, 
                num_folds=random.randint(1, 3), 
                fold_amplitude=fold_level
            )
            
        # 2. Perspective Warp & Background Blending
        if perspective_level > 0 or bg_type != "white":
            img, current_ann = ReceiptAugmenter.apply_perspective_warp(
                img, current_ann, 
                max_distortion=perspective_level,
                bg_type=bg_type
            )
            
        # 3. Bóng mờ & Đèn flash
        if shadow_level > 0 or flash_level > 0:
            img = ReceiptAugmenter.apply_shadows_and_flashlight(
                img, 
                shadow_intensity=shadow_level, 
                flash_intensity=flash_level
            )
            
        # 4. Phai màu & Vệt đầu in nhiệt
        if fade_level > 0 or streak_level > 0:
            img = ReceiptAugmenter.apply_thermal_fading_and_streaks(
                img,
                fade_intensity=fade_level,
                streak_probability=streak_level
            )
            
        return img, current_ann
