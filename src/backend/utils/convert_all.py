# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script chuyển đổi nhãn mồi từ dạng 'predictions' sang 'annotations' cho tất cả các thành viên.
"""

import os
import json
import sys

# Đảm bảo hiển thị tiếng Việt không bị lỗi font trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

folders = ["task_dai", "task_cam", "task_vuong"]
files_to_convert = ["pre_labels.json", "pre_labels_local.json"]

print("=== BẮT ĐẦU CHUYỂN ĐỔI 'PREDICTIONS' SANG 'ANNOTATIONS' ===")

for folder in folders:
    if not os.path.exists(folder):
        print(f"[WARNING] Thư mục {folder} không tồn tại. Bỏ qua.")
        continue
        
    for filename in files_to_convert:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"\n[INFO] Đang xử lý file: {filepath}...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            modified = False
            for task in data:
                if "predictions" in task:
                    # Chuyển đổi predictions thành annotations
                    annotations = []
                    for pred in task["predictions"]:
                        annotations.append({
                            "result": pred.get("result", [])
                        })
                    task["annotations"] = annotations
                    # Xóa key predictions cũ
                    del task["predictions"]
                    modified = True
            
            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"[SUCCESS] Đã chuyển đổi thành công: {filepath}")
            else:
                print(f"[INFO] File {filepath} đã ở dạng 'annotations' rồi (không cần chuyển).")
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý file {filepath}: {str(e)}")

print("\n=== HOÀN TẤT CHUYỂN ĐỔI ===")
