import os
import json

print("=========================================")
print("🚀 SCRIPT NẠP DATA DÀNH RIÊNG CHO RUNPOD (LINUX) 🚀")
print("=========================================")

data_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(data_dir):
    print("❌ Lỗi: Bạn phải để file script này NẰM TRONG thư mục LLaMA-Factory nhé!")
    exit()

dataset_dir = input("👉 Hãy dán đường dẫn thư mục FINAL_SPLIT_DATASET trên RunPod vào đây (Ví dụ: /workspace/FINAL_SPLIT_DATASET): ").strip()
dataset_dir = dataset_dir.strip('"').strip("'")

splits = {"train": "train.jsonl", "val": "val.jsonl"}
registered_datasets = {}

for split_name, file_name in splits.items():
    jsonl_path = os.path.join(dataset_dir, split_name, file_name)
    
    if not os.path.exists(jsonl_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy {jsonl_path}. Bỏ qua tập {split_name}.")
        continue

    print(f"🔄 Đang xử lý tập {split_name} (Hơi lâu xíu, chịu khó đợi)...")
    llama_dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            
            # Tạo đường dẫn ảnh tuyệt đối cho Linux
            abs_images = []
            for img in data.get('images', []):
                # img có dạng "images/ten_file.jpg"
                img_path = os.path.join(dataset_dir, split_name, img)
                abs_images.append(img_path.replace("\\", "/"))
            data['images'] = abs_images
            llama_dataset.append(data)
            
    # Lưu vào thư mục LLaMA-Factory/data
    out_name = f"minicpmv_{split_name}.json"
    out_path = os.path.join(data_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(llama_dataset, f, ensure_ascii=False, indent=2)
        
    registered_datasets[f"minicpmv_{split_name}"] = out_name

# Đăng ký danh tính vào hệ thống của LLaMA-Factory
if registered_datasets:
    info_path = os.path.join(data_dir, "dataset_info.json")
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    except:
        dataset_info = {}

    for ds_key, file_name in registered_datasets.items():
        dataset_info[ds_key] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"✅ QUÁ TUYỆT VỜI! Đã nạp thành công các tập: {list(registered_datasets.keys())}.")
    print("👉 Giờ bạn gõ 'llamafactory-cli webui' để chiến thôi!")
else:
    print("❌ Không nạp được tập data nào. Anh/chị kiểm tra lại xem đường dẫn đã trỏ đúng vào FINAL_SPLIT_DATASET chưa nhé!")
