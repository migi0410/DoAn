# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script fine-tune YOLO11s-Pose trên tập dữ liệu ảnh thực tế MC-OCR
      giúp mô hình tối ưu hóa độ chính xác phát hiện 4 góc trong môi trường thực tế
      (nền bàn phức tạp, giấy nhăn, bóng đổ và góc nghiêng đa dạng).
"""

import os
import sys
import argparse
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s-Pose trên ảnh thực tế MC-OCR")
    parser.add_argument("--model", default="backend/weights/yolo11s_pose_crop.pt", help="Checkpoint xuất phát (mặc định model production)")
    parser.add_argument("--data", default="dataset_yolo_real_mcocr/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện (mặc định 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train (mặc định 640)")
    parser.add_argument("--device", default="0", help="GPU index (0) hoặc cpu")
    parser.add_argument("--project", default="runs/pose", help="Thư mục lưu kết quả train")
    parser.add_argument("--name", default="real_mcocr_yolo11s", help="Tên phiên train")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"[LỖI] Không tìm thấy file data.yaml tại: {data_path}")
        sys.exit(1)

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file model tại: {model_path}")
        sys.exit(1)

    print("==================================================================")
    print("      BẮT ĐẦU FINE-TUNE YOLO11s-POSE TRÊN ẢNH THỰC TẾ MC-OCR      ")
    print("==================================================================")
    print(f"Model cơ sở:    {model_path}")
    print(f"Cấu hình data:   {data_path}")
    print(f"Số epochs:       {args.epochs}")
    print(f"Batch size:      {args.batch}")
    print(f"Kích thước ảnh:  {args.imgsz}x{args.imgsz}")
    print(f"Thiết bị:        {args.device}")
    print(f"Thư mục lưu:     {os.path.join(args.project, args.name)}")
    print("==================================================================\n")

    # Load model
    model = YOLO(model_path)

    # Huấn luyện fine-tune với cấu hình tối ưu cho corner keypoint detection
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        # Fine-tune learning rate
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,
        # Augmentations bảo toàn thứ tự 4 góc: TL -> TR -> BR -> BL
        flipud=0.0,
        fliplr=0.0,
        degrees=15.0,
        shear=4.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        # Lưu và đánh giá
        plots=True,
        save=True,
        save_period=10,
        workers=4,
        patience=15,
        verbose=True
    )

    best_weight = os.path.join(args.project, args.name, "weights", "best.pt")
    print("\n==================================================================")
    print("   FINE-TUNING HOÀN TẤT THÀNH CÔNG!")
    print(f"   Trọng số tối ưu nhất được lưu tại: {best_weight}")
    print("==================================================================")

if __name__ == "__main__":
    main()
