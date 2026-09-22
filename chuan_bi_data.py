import os
import json

print("=========================================")
dataset_dir = input("👉 Hãy Copy & Dán đường dẫn thư mục Dataset vào đây (Ví dụ: C:\\Users\\...\\Desktop\\Dataset): ").strip()
dataset_dir = dataset_dir.strip('"').strip("'")

train_jsonl = os.path.join(dataset_dir, "train.jsonl")
if not os.path.exists(train_jsonl):
    print(f"❌ Không tìm thấy file train.jsonl trong thư mục: {dataset_dir}")
    exit()

print("🔄 Đang chuyển đổi và nạp dữ liệu cho LLaMA-Factory...")
llama_dataset = []
with open(train_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line.strip())
        
        # Sửa đường dẫn ảnh thành đường dẫn tuyệt đối trên máy Windows
        abs_images = []
        for img in data.get('images', []):
            abs_images.append(os.path.join(dataset_dir, img.replace("\\", "/")))
        data['images'] = abs_images
        llama_dataset.append(data)

# Lưu vào thư mục data của LLaMA-Factory
data_dir = os.path.join(os.getcwd(), "data")
dataset_path = os.path.join(data_dir, "minicpmv_dataset.json")
with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(llama_dataset, f, ensure_ascii=False, indent=2)

# Đăng ký dataset vào hệ thống
info_path = os.path.join(data_dir, "dataset_info.json")
with open(info_path, 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)

dataset_info["minicpmv_data"] = {
    "file_name": "minicpmv_dataset.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
}
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print("✅ THÀNH CÔNG! Đã nạp Data xong. Bạn có thể bật WebUI để Train!")
