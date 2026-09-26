import os
from dotenv import load_dotenv
from supabase import create_client, Client
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rdtyboaufaqbvmnombwp.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJkdHlib2F1ZmFxYnZtbm9tYndwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA0Mzk2MzksImV4cCI6MjEwNjAxNTYzOX0.O4KwKbDzbgRIczD0kIJyiQXzypCWwMMEGEPVCr_Y-vs")

try:
    if SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        supabase = None
except Exception as e:
    print("Supabase init error:", e)
    supabase = None

def upload_image_to_supabase(local_file_path: str, user_id: str, receipt_id: str) -> str:
    if not supabase:
        print("Cảnh báo: Supabase chưa được cấu hình. Bỏ qua upload.")
        return ""
        
    try:
        bucket_name = "receipts"
        file_ext = os.path.splitext(local_file_path)[1]
        file_name = f"{user_id}/{receipt_id}{file_ext}"
        
        with open(local_file_path, "rb") as f:
            res = supabase.storage.from_(bucket_name).upload(
                path=file_name,
                file=f,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
            
        public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
        return public_url
    except Exception as e:
        print("Lỗi khi upload ảnh lên Supabase:", e)
        return ""
