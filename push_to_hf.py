import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN", "")

BASE_DIR = r"C:\Users\Admin\OneDrive\DoAn"
IMAGES_DIR = os.path.join(BASE_DIR, "FINAL_RUNPOD_DATASET", "images")
TRAIN_JSONL = os.path.join(BASE_DIR, "FINAL_RUNPOD_DATASET", "train.jsonl")
BBOX_JSON = os.path.join(BASE_DIR, "FINAL_BBOX_DATASET_V3.json")

def upload_to_huggingface():
    print("🔑 Đang đăng nhập vào Hugging Face...")
    login(token=HF_TOKEN)
    
    api = HfApi()
    username = api.whoami(token=HF_TOKEN)["name"]
    DATASET_REPO_ID = f"{username}/invoice-vlm-dataset"
    
    print(f"📦 Đang khởi tạo kho dữ liệu: {DATASET_REPO_ID}")
    # Tạo repo nếu chưa tồn tại (private=True để bảo mật nội bộ team)
    api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    
    print("⏳ Đang tải lên file Bounding Box JSON (Dành cho EDA)...")
    if os.path.exists(BBOX_JSON):
        api.upload_file(
            path_or_fileobj=BBOX_JSON,
            path_in_repo="FINAL_BBOX_DATASET_V3.json",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset"
        )
        print("✅ Đã tải lên FINAL_BBOX_DATASET_V3.json")
    else:
        print(f"⚠️ KhÔng tìm thấy {BBOX_JSON}")
    
    print("⏳ Đang tải lên file Train JSONL (Dành cho Training VLM)...")
    if os.path.exists(TRAIN_JSONL):
        api.upload_file(
            path_or_fileobj=TRAIN_JSONL,
            path_in_repo="train.jsonl",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset"
        )
        print("✅ Đã tải lên train.jsonl")
    else:
        print(f"⚠️ Không tìm thấy {TRAIN_JSONL}")
        
    print("Bước 4: Đang tải lên file images.tar (Khoảng 5.6GB, có thể mất khá nhiều thời gian)...")
    TAR_FILE = os.path.join(BASE_DIR, "FINAL_RUNPOD_DATASET", "images.tar")
    if os.path.exists(TAR_FILE):
        api.upload_file(
            path_or_fileobj=TAR_FILE,
            path_in_repo="images.tar",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset"
        )
        print("Done. Đã tải lên file images.tar thành công!")
    else:
        print(f"Lỗi: Không tìm thấy {TAR_FILE}")

    print(f"\n🎉 HOÀN TẤT! Dữ liệu của bạn đã nằm an toàn trên Hugging Face.")
    print(f"👉 Link truy cập: https://huggingface.co/datasets/{DATASET_REPO_ID}")
    print("Mọi người trong team (được cấp quyền) giờ đây có thể dùng thư viện `datasets` để load data về làm việc!")

if __name__ == "__main__":
    upload_to_huggingface()
