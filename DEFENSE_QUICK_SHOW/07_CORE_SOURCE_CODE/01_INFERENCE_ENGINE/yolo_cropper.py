# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Module sản xuất (Production Module): YOLO11s-Pose Preprocessing Engine
      - Nhận diện 4 góc mép giấy hóa đơn bằng mạng nơ-ron sâu YOLO11s-Pose
      - Tự động phục hồi góc khuyết bằng Parallelogram Vector Completion
      - Nắn phẳng phối cảnh (Perspective Dewarping) về góc 0.0°
      - Cân bằng tương phản thích ứng cục bộ CLAHE (LAB color space)
"""

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YOLOInvoiceCropper:
    _instance = None

    def __new__(cls, weights_path=None):
        if cls._instance is None:
            cls._instance = super(YOLOInvoiceCropper, cls).__new__(cls)
            cls._instance._init_model(weights_path)
        return cls._instance

    def _init_model(self, weights_path=None):
        # Xác định đường dẫn trọng số theo thứ tự ưu tiên
        candidate_paths = [
            weights_path,
            os.path.join(os.path.dirname(__file__), "weights", "yolo11s_pose_crop.pt"),
            os.path.join(os.path.dirname(__file__), "..", "trained_models", "yolo11s_pose_crop.pt"),
            os.path.join(os.path.dirname(__file__), "..", "runs", "pose", "invoice_crop", "weights", "best.pt"),
            "/home/haderax/DoAn/backend/weights/yolo11s_pose_crop.pt",
            "/home/haderax/DoAn/yolo_train/runs/pose/runs/pose/invoice_crop_yolo11s-2/weights/best.pt"
        ]

        resolved_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        if not resolved_path:
            raise FileNotFoundError(
                f"[YOLOInvoiceCropper] Không tìm thấy file trọng số YOLO11s-Pose tại các đường dẫn: {candidate_paths}"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[YOLOInvoiceCropper] Nạp mô hình từ: {resolved_path} (Thiết bị: {self.device})")
        self.model = YOLO(resolved_path)
        self.model.to(self.device)
        self.weights_path = resolved_path

        # Warm-up model 1 lần để tăng tốc độ suy luận lần đầu
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            print("[YOLOInvoiceCropper] Khởi tạo & Warm-up thành công!")
        except Exception as e:
            print(f"[YOLOInvoiceCropper] Cảnh báo warm-up: {e}")

    @staticmethod
    def fix_missing_corners(kpts, bbox=None):
        """
        Bù góc khuyết thông minh dựa trên quan hệ hình học hình bình hành
        hoặc fallback về 4 góc của bounding box.
        """
        pts = np.copy(kpts)
        zeros = [i for i in range(4) if np.all(pts[i] == 0) or (pts[i][0] == 0 and pts[i][1] == 0)]
        if len(zeros) == 1:
            z = zeros[0]
            if z == 3:   # BL thiếu -> BL = TL + (BR - TR)
                pts[3] = pts[0] + (pts[2] - pts[1])
            elif z == 2: # BR thiếu -> BR = TR + (BL - TL)
                pts[2] = pts[1] + (pts[3] - pts[0])
            elif z == 1: # TR thiếu -> TR = TL + (BR - BL)
                pts[1] = pts[0] + (pts[2] - pts[3])
            elif z == 0: # TL thiếu -> TL = TR + (BL - BR)
                pts[0] = pts[1] + (pts[3] - pts[2])
        elif len(zeros) > 1 and bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            pts = np.array([
                [xmin, ymin], # TL
                [xmax, ymin], # TR
                [xmax, ymax], # BR
                [xmin, ymax]  # BL
            ], dtype=np.float32)
        return pts

    @staticmethod
    def crop_and_dewarp(image, keypoints):
        """
        Thực hiện phép biến đổi phối cảnh (Perspective Transform) nắn thẳng hóa đơn.
        """
        pts_src = np.float32(keypoints)
        width_top = np.linalg.norm(pts_src[1] - pts_src[0])
        width_bottom = np.linalg.norm(pts_src[2] - pts_src[3])
        max_w = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(pts_src[3] - pts_src[0])
        height_right = np.linalg.norm(pts_src[2] - pts_src[1])
        max_h = int(max(height_left, height_right))

        if max_w <= 10 or max_h <= 10:
            return image.copy()

        pts_dst = np.float32([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
        return warped

    @staticmethod
    def enhance_illumination(image):
        """
        Cân bằng tương phản thích ứng cục bộ CLAHE trên không gian màu LAB.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process(self, img: np.ndarray, conf_thresh: float = 0.15) -> dict:
        """
        Thực hiện chu trình tiền xử lý hoàn chỉnh trên 1 ảnh BGR:
        1. Suy luận YOLO11s-Pose
        2. Lấy tọa độ 4 góc & Bù góc khuyết
        3. Tính góc nghiêng (skew angle)
        4. Nắn phẳng phối cảnh (Dewarping)
        5. Cân bằng sáng CLAHE
        """
        h0, w0 = img.shape[:2]
        t_start = time.perf_counter()

        preds = self.model(img, conf=conf_thresh, verbose=False)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

        pred_kpts = None
        if len(preds) > 0 and preds[0].boxes is not None and len(preds[0].boxes) > 0:
            bbox_px = preds[0].boxes.xyxy.cpu().numpy()[0]
            if preds[0].keypoints is not None and len(preds[0].keypoints.xy) > 0:
                raw_kpts = preds[0].keypoints.xy.cpu().numpy()[0]
                pred_kpts = self.fix_missing_corners(raw_kpts, bbox=bbox_px)
            else:
                xmin, ymin, xmax, ymax = bbox_px
                pred_kpts = np.array([
                    [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                ], dtype=np.float32)

        if pred_kpts is None:
            # Fallback toàn khung ảnh nếu không phát hiện
            pred_kpts = np.array([
                [0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]
            ], dtype=np.float32)

        # 1. Tính góc nghiêng cạnh trên P1 -> P2
        dx = pred_kpts[1][0] - pred_kpts[0][0]
        dy = pred_kpts[1][1] - pred_kpts[0][1]
        skew_angle = round(float(np.degrees(np.arctan2(dy, dx))), 1)

        # 2. Tọa độ chuẩn hóa theo tỷ lệ phần trăm (0 - 100%) cho Frontend
        corners_ui = [
            {"label": "P1 (Top-Left)", "x": round(float(pred_kpts[0][0] / w0 * 100), 1), "y": round(float(pred_kpts[0][1] / h0 * 100), 1)},
            {"label": "P2 (Top-Right)", "x": round(float(pred_kpts[1][0] / w0 * 100), 1), "y": round(float(pred_kpts[1][1] / h0 * 100), 1)},
            {"label": "P3 (Bottom-Right)", "x": round(float(pred_kpts[2][0] / w0 * 100), 1), "y": round(float(pred_kpts[2][1] / h0 * 100), 1)},
            {"label": "P4 (Bottom-Left)", "x": round(float(pred_kpts[3][0] / w0 * 100), 1), "y": round(float(pred_kpts[3][1] / h0 * 100), 1)},
        ]

        # 3. Nắn phẳng phối cảnh & Cân bằng sáng
        dewarped = self.crop_and_dewarp(img, pred_kpts)
        enhanced = self.enhance_illumination(dewarped)
        h1, w1 = enhanced.shape[:2]

        return {
            "corners": corners_ui,
            "corners_px": pred_kpts,
            "skew_angle": skew_angle,
            "latency_ms": latency_ms,
            "dewarped": dewarped,
            "enhanced": enhanced,
            "w_orig": w0,
            "h_orig": h0,
            "w_prep": w1,
            "h_prep": h1,
            "device": self.device,
            "method": f"YOLO11s-Pose (Keypoint Corner Detection & Perspective Dewarp - {self.device})"
        }

_CROPPER_SINGLETON = None

def get_yolo_cropper(weights_path=None) -> YOLOInvoiceCropper:
    global _CROPPER_SINGLETON
    if _CROPPER_SINGLETON is None:
        _CROPPER_SINGLETON = YOLOInvoiceCropper(weights_path)
    return _CROPPER_SINGLETON
