# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE (Trích xuất thông tin hóa đơn tiếng Việt)
Mô tả: Script gộp các file JSON gán nhãn hoàn chỉnh từ các thành viên thành master dataset.
Tác giả: Dũng (Leader)
"""

import os
import sys
import json
import glob

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Định nghĩa các thành viên và thư mục tương ứng
MEMBERS = ["dai", "cam", "vuong"]
ORIGINAL_IMAGE_DIRS = ["saigon_coop_flat_images", "vincommerce_flat_images"]
OUTPUT_FILE = "master_dataset.json"

def find_member_annotation_file(member):
    """
    Tự động tìm kiếm file JSON kết quả gán nhãn của thành viên.
    Thứ tự tìm kiếm:
    1. annotated_{member}.json ở thư mục gốc.
    2. task_{member}/annotated_{member}.json.
    3. Bất kỳ file JSON nào trong task_{member}/ ngoại trừ pre_labels.json và pre_labels_local.json.
    """
    # 1. Tìm ở thư mục gốc
    root_path = f"annotated_{member}.json"
    if os.path.exists(root_path):
        return root_path
        
    # 2. Tìm trong thư mục task tương ứng
    task_dir = f"task_{member}"
    task_path = os.path.join(task_dir, f"annotated_{member}.json")
    if os.path.exists(task_path):
        return task_path
        
    # 3. Quét tất cả file JSON khác trong thư mục task
    if os.path.exists(task_dir):
        json_files = glob.glob(os.path.join(task_dir, "*.json"))
        for jf in json_files:
            filename = os.path.basename(jf)
            if filename not in ["pre_labels.json", "pre_labels_local.json"]:
                return jf
                
    return None

def extract_filename_from_url(url_path):
    """
    Trích xuất tên file ảnh từ đường dẫn hoặc URL của Label Studio.
    Ví dụ:
    - "/data/local-files/?d=C:/Users/Dai/task_dai/images/abc.jpg" -> "abc.jpg"
    - "images/abc.jpg" -> "abc.jpg"
    """
    if not url_path:
        return ""
    
    # Nếu là URL local file của Label Studio
    if "/data/local-files/?d=" in url_path:
        # Lấy phần sau "?d="
        path_part = url_path.split("?d=")[-1]
        return os.path.basename(path_part)
        
    # Mặc định lấy tên file cuối cùng
    return os.path.basename(url_path)

def resolve_original_image_path(filename, search_dirs):
    """
    Tìm lại đường dẫn tương đối gốc của ảnh trong các thư mục saigon_coop hoặc vincommerce.
    """
    for d in search_dirs:
        # Kiểm tra sự tồn tại của ảnh trong thư mục gốc
        possible_path = os.path.join(d, filename)
        if os.path.exists(possible_path):
            # Trả về định dạng đường dẫn sử dụng dấu gạch chéo xuôi cho tương thích đa nền tảng
            return possible_path.replace("\\", "/")
            
    # Nếu không tìm thấy, trả về đường dẫn tương đối mặc định
    return f"images/{filename}"

def main():
    print("=== PIPELINE GÁN NHÃN BÁN TỰ ĐỘNG - GỘP DỮ LIỆU GÁN NHÃN ===")
    
    master_tasks = []
    global_task_id = 1
    total_images_processed = 0
    total_annotations_count = 0
    
    for member in MEMBERS:
        print(f"\n[INFO] Đang tìm dữ liệu của thành viên: {member.upper()}")
        file_path = find_member_annotation_file(member)
        
        if not file_path:
            print(f"[WARNING] Không tìm thấy file JSON gán nhãn của {member.upper()}. Bỏ qua thành viên này.")
            continue
            
        print(f"[INFO] Tìm thấy file: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
                
            if not isinstance(tasks, list):
                print(f"[ERROR] Định dạng dữ liệu trong file '{file_path}' không phải là danh sách (list). Bỏ qua.")
                continue
                
            print(f"[INFO] File chứa {len(tasks)} tasks.")
            member_annotated_count = 0
            
            for task_idx, task in enumerate(tasks):
                # Kiểm tra xem task đã được gán nhãn hay chưa (tránh mất dữ liệu chưa hoàn thiện)
                annotations = task.get("annotations", [])
                
                # Trong Label Studio, task đã review sẽ có phần tử trong danh sách annotations
                if not annotations:
                    # Nếu thành viên chưa sửa đổi gì, có thể lấy nhãn mồi (predictions) hoặc skip
                    # Ở đây chúng ta sẽ cảnh báo nhưng vẫn giữ lại để tránh mất ảnh
                    print(f"  - Task {task.get('id', task_idx)} chưa có annotation từ người dùng. Sử dụng nhãn dự đoán ban đầu.")
                    
                # Trích xuất tên file ảnh thực tế
                raw_image_path = task.get("data", {}).get("image", "")
                filename = extract_filename_from_url(raw_image_path)
                
                if not filename:
                    print(f"  - [WARNING] Task {task.get('id', task_idx)} không có thông tin ảnh. Bỏ qua.")
                    continue
                    
                # Tìm lại đường dẫn gốc của ảnh
                resolved_img_path = resolve_original_image_path(filename, ORIGINAL_IMAGE_DIRS)
                
                # Tạo bản ghi task sạch sẽ cho master dataset
                merged_task = {
                    "id": global_task_id,
                    "data": {
                        "image": resolved_img_path
                    },
                    # Lưu lại cả phần annotations (nhãn người sửa) và predictions (nhãn mồi) nếu có
                    "annotations": annotations,
                }
                
                if "predictions" in task:
                    merged_task["predictions"] = task["predictions"]
                    
                master_tasks.append(merged_task)
                global_task_id += 1
                total_images_processed += 1
                
                if annotations:
                    member_annotated_count += 1
                    total_annotations_count += 1
                    
            print(f"[SUCCESS] Đã gộp thành công {len(tasks)} tasks từ {member.upper()} (Đã gán nhãn: {member_annotated_count}/{len(tasks)}).")
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi đọc file '{file_path}': {str(e)}")
            
    # Xuất ra master_dataset.json
    if master_tasks:
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(master_tasks, f, indent=2, ensure_ascii=False)
            
            print("\n=======================================================")
            print(f"[SUCCESS] Đã tạo thành công Master Dataset tại: {OUTPUT_FILE}")
            print(f"Tổng số ảnh gộp được: {total_images_processed}")
            print(f"Tổng số ảnh có nhãn người sửa (annotations): {total_annotations_count}")
            print("Sẵn sàng cho các bước convert sang định dạng training LayoutLM.")
            print("=======================================================")
        except Exception as e:
            print(f"\n[ERROR] Không thể ghi file output '{OUTPUT_FILE}'. Chi tiết: {str(e)}")
    else:
        print("\n[WARNING] Không thu thập được dữ liệu nào từ các thành viên. Không tạo file master_dataset.json.")

if __name__ == "__main__":
    main()
