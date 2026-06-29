# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE (Trích xuất thông tin hóa đơn tiếng Việt)
Mô tả: Script mồi nhãn thô sử dụng PaddleOCR và tự động chia đều việc gán nhãn cho các thành viên.
Tác giả: Dũng (Leader)
"""

import os
import sys

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thiết lập môi trường để tránh lỗi protobuf với PaddlePaddle
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import shutil
import glob
import numpy as np

# Patch np.sctypes which was removed in NumPy 2.0 (needed by imgaug / paddleocr)
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, str, bytes]
    }

# Disable MKLDNN to avoid "could not create a primitive" crash on certain image shapes on CPU
os.environ["FLAGS_use_mkldnn"] = "0"

from PIL import Image
from paddleocr import PaddleOCR

# Cấu hình đường dẫn đầu vào và đầu ra
INPUT_DIRS = ["mcocr/train_images/train_images", "mcocr/val_images/val_images"]
MEMBERS = ["dai", "cam", "vuong"]
OUTPUT_PREFIX = "task_"

def get_all_images(dirs):
    """Quét toàn bộ ảnh từ các thư mục đầu vào."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_images = {}
    
    for d in dirs:
        if not os.path.exists(d):
            print(f"[WARNING] Thư mục '{d}' không tồn tại. Bỏ qua.")
            continue
        
        images_in_dir = []
        for ext in extensions:
            images_in_dir.extend(glob.glob(os.path.join(d, ext)))
        
        # Loại bỏ trùng lặp (nhất là trên Windows do không phân biệt chữ hoa/thường)
        images_in_dir = list(set(os.path.normpath(p).replace("\\", "/") for p in images_in_dir))
        
        # Sắp xếp để đảm bảo tính nhất quán
        images_in_dir.sort()
        all_images[d] = images_in_dir
        print(f"[INFO] Tìm thấy {len(images_in_dir)} ảnh trong thư mục '{d}'.")
        
    return all_images

def split_dataset(all_images_dict, members):
    """Chia đều ảnh của từng thư mục cho các thành viên để đảm bảo cân bằng dữ liệu."""
    member_tasks = {member: [] for member in members}
    num_members = len(members)
    
    for dir_name, images in all_images_dict.items():
        # Chia đều danh sách ảnh của thư mục này
        for idx, img_path in enumerate(images):
            assigned_member = members[idx % num_members]
            member_tasks[assigned_member].append(img_path)
            
    # In ra số lượng công việc được phân chia
    print("\n--- BẢNG PHÂN CHIA CÔNG VIỆC ---")
    for member, tasks in member_tasks.items():
        print(f"Thành viên: {member.upper()} | Số lượng ảnh: {len(tasks)}")
    print("---------------------------------\n")
    
    return member_tasks

def process_and_generate_prelabels(member, img_paths, ocr):
    """Chạy OCR, chuyển đổi tọa độ sang %, copy ảnh và lưu pre_labels.json. Hỗ trợ resume từ lượt chạy trước."""
    member_dir = f"{OUTPUT_PREFIX}{member}"
    images_dir = os.path.join(member_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    output_json_path = os.path.join(member_dir, "pre_labels.json")
    
    # Hỗ trợ Resume logic: đọc các task đã thành công từ lượt chạy trước
    existing_tasks = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for task in loaded:
                    img_val = task["data"]["image"]
                    filename = os.path.basename(img_val)
                    existing_tasks[filename] = task
            print(f"[INFO] Tìm thấy {len(existing_tasks)} ảnh đã gán nhãn thô trước đó cho {member.upper()}. Sẽ bỏ qua chúng.")
        except Exception as e:
            print(f"[WARNING] Không thể đọc file pre_labels.json cũ: {str(e)}")
            
    pre_labels_tasks = []
    success_count = 0
    fail_count = 0
    
    print(f"\n[START] Bắt đầu xử lý cho thành viên: {member.upper()}")
    
    for idx, orig_path in enumerate(img_paths):
        filename = os.path.basename(orig_path)
        dest_path = os.path.join(images_dir, filename)
        
        # Nếu ảnh đã được xử lý và lưu trong pre_labels.json trước đó, ta tái sử dụng
        if filename in existing_tasks:
            pre_labels_tasks.append(existing_tasks[filename])
            success_count += 1
            continue
            
        print(f"[{member.upper()}] Xử lý ảnh {idx + 1}/{len(img_paths)}: {filename}")
        
        try:
            # 1. Đọc kích thước ảnh gốc
            with Image.open(orig_path) as img:
                width, height = img.size
            
            # 2. Chạy PaddleOCR
            # ocr.ocr trả về một list các kết quả, mỗi kết quả chứa box và text
            result = ocr.ocr(orig_path, cls=True)
            
            # Xử lý trường hợp OCR không nhận diện được chữ nào hoặc bị rỗng
            if not result or result[0] is None:
                ocr_regions = []
            else:
                ocr_regions = result[0]
                
            predictions_result = []
            
            # 3. Chuyển đổi tọa độ box sang % của Label Studio
            for box_idx, region in enumerate(ocr_regions):
                box = region[0]        # [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                text_info = region[1]  # (text, confidence)
                text = text_info[0]
                
                # Tọa độ 4 góc của bounding box
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                
                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # Tính phần trăm
                x_pct = (x_min / width) * 100
                y_pct = (y_min / height) * 100
                w_pct = (box_w / width) * 100
                h_pct = (box_h / height) * 100
                
                # Giới hạn trong khoảng [0, 100] tránh lỗi hiển thị ngoài màn hình
                x_pct = max(0.0, min(100.0, x_pct))
                y_pct = max(0.0, min(100.0, y_pct))
                w_pct = max(0.0, min(100.0 - x_pct, w_pct))
                h_pct = max(0.0, min(100.0 - y_pct, h_pct))
                
                region_id = f"region_{box_idx}"
                
                # Tạo Rectangle label (mồi nhãn mặc định là OTHER)
                rect_label = {
                    "id": region_id,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": round(x_pct, 4),
                        "y": round(y_pct, 4),
                        "width": round(w_pct, 4),
                        "height": round(h_pct, 4),
                        "rotation": 0,
                        "rectanglelabels": ["OTHER"]
                    }
                }
                
                # Tạo Textarea transcription (chứa text thô nhận diện)
                text_area = {
                    "id": region_id,
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea",
                    "value": {
                        "x": round(x_pct, 4),
                        "y": round(y_pct, 4),
                        "width": round(w_pct, 4),
                        "height": round(h_pct, 4),
                        "rotation": 0,
                        "text": [text]
                    }
                }
                
                predictions_result.extend([rect_label, text_area])
                
            # 4. Copy ảnh vật lý vào thư mục đích của thành viên
            shutil.copy2(orig_path, dest_path)
            
            # 5. Tạo cấu trúc task của Label Studio (sử dụng đường dẫn tương đối ban đầu)
            # Đường dẫn này sẽ được các thành viên cập nhật thành tuyệt đối cục bộ sau
            task_data = {
                "data": {
                    "image": f"images/{filename}"
                },
                "annotations": [
                    {
                        "result": predictions_result
                    }
                ]
            }
            
            pre_labels_tasks.append(task_data)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Không thể xử lý ảnh '{orig_path}'. Lỗi: {str(e)}")
            fail_count += 1
            
    # 6. Ghi ra file JSON pre_labels.json
    output_json_path = os.path.join(member_dir, "pre_labels.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(pre_labels_tasks, f, indent=2, ensure_ascii=False)
        
    print(f"[DONE] Hoàn thành cho {member.upper()}. Thành công: {success_count}, Thất bại: {fail_count}")
    print(f"Đã lưu kết quả tại: {output_json_path}")

def main():
    print("=== PIPELINE GÁN NHÃN BÁN TỰ ĐỘNG - KHỞI TẠO & CHIA VIỆC ===")
    
    # Bước 1: Quét toàn bộ ảnh
    all_images = get_all_images(INPUT_DIRS)
    
    # Hỗ trợ cấu hình số lượng giới hạn ảnh chạy thử nghiệm qua biến môi trường
    limit_env = os.environ.get("LIMIT_IMAGES")
    if limit_env:
        try:
            limit_val = int(limit_env)
            print(f"[INFO] Chế độ chạy thử nghiệm: Giới hạn tối đa {limit_val} ảnh mỗi thư mục.")
            for d in all_images:
                all_images[d] = all_images[d][:limit_val]
        except ValueError:
            pass
            
    total_images = sum(len(imgs) for imgs in all_images.values())
    
    if total_images == 0:
        print("[ERROR] Không tìm thấy ảnh nào trong các thư mục đầu vào. Vui lòng kiểm tra lại cấu trúc thư mục.")
        return
        
    # Bước 2: Chia đều ảnh cho các thành viên
    member_tasks = split_dataset(all_images, MEMBERS)
    
    # Bước 3: Khởi tạo PaddleOCR
    print("[INFO] Đang khởi tạo PaddleOCR (vi)...")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False, use_mkldnn=False)
        print("[INFO] Khởi tạo PaddleOCR thành công.")
    except Exception as e:
        print(f"[ERROR] Không thể khởi tạo PaddleOCR. Chi tiết lỗi: {str(e)}")
        return
        
    # Bước 4: Xử lý và phân phối cho từng thành viên
    for member in MEMBERS:
        process_and_generate_prelabels(member, member_tasks[member], ocr)
        
    print("\n=======================================================")
    print("[SUCCESS] Pipeline hoàn tất. Vui lòng kiểm tra các thư mục task_*. ")
    print("Hãy gửi các thư mục tương ứng cho Đại, Cẩm, Vương kèm file hướng dẫn README.md.")

if __name__ == "__main__":
    main()
