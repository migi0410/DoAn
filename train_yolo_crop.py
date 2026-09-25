# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script huấn luyện mô hình YOLOv8 / YOLO11 Pose để phát hiện 4 góc hóa đơn
      và cắt rìa / nắn thẳng phối cảnh (Perspective Dewarping).
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO Pose cắt góc hóa đơn")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Checkpoint ban đầu (yolov8n-pose.pt hoặc yolo11n-pose.pt)")
    parser.add_argument("--data", default="dataset_yolo_crop/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện (mặc định 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định 16 phù hợp RTX 3050)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train (mặc định 640)")
    parser.add_argument("--device", default="0", help="GPU index (0) hoặc cpu")
    parser.add_argument("--project", default="runs/pose", help="Thư mục lưu kết quả train")
    parser.add_argument("--name", default="invoice_crop", help="Tên phiên train")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"[LỖI] Không tìm thấy file data.yaml tại: {data_path}")
        sys.exit(1)

    print("=========================================================")
    print("      BẮT ĐẦU HUẤN LUYỆN YOLO POSE (CẮT GÓC HÓA ĐƠN)     ")
    print("=========================================================")
    print(f"Model cơ sở:  {args.model}")
    print(f"Cấu hình data: {data_path}")
    print(f"Số epochs:     {args.epochs}")
    print(f"Batch size:    {args.batch}")
    print(f"Kích thước:    {args.imgsz}x{args.imgsz}")
    print(f"Thiết bị:      {args.device}")
    print("=========================================================\n")

    # Load model pre-trained
    model = YOLO(args.model)

    # Bắt đầu huấn luyện
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        save=True,
        save_period=10,
        workers=4,
        patience=15,
        verbose=True
    )

    best_weight = os.path.join(args.project, args.name, "weights", "best.pt")
    print("\n=========================================================")
    print("   HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"   Trọng số tối ưu nhất được lưu tại: {best_weight}")
    print("=========================================================")

if __name__ == "__main__":
    main()
