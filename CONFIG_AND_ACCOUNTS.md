# TÀI LIỆU CẤU HÌNH HỆ THỐNG VÀ DANH SÁCH TÀI KHOẢN
### Đề tài: AVIR-KIE (Phân tích bố cục & Trích xuất thông tin hóa đơn tiếng Việt)
**Mã đề tài:** `SU26AI50` | **Mã nhóm:** `GSU26AI01`  
**Bộ môn:** Trí tuệ Nhân tạo (AI) – Đại học FPT TP. Hồ Chí Minh  
**Giảng viên hướng dẫn:** Thầy Nguyễn Hồng Hải (`hainh51@fe.edu.vn`)

---

## 1. CẤU HÌNH CÁC THÀNH PHẦN HỆ THỐNG (PORTS & ENDPOINTS)

Toàn bộ hệ thống được xây dựng theo kiến trúc Microservices phân tán, tách biệt giữa giao diện người dùng, nghiệp vụ điều phối và máy chủ tính toán GPU:

| Thành phần hệ thống | Công nghệ / Framework | Port mặc định | URL truy cập nội bộ | Chức năng chính |
| :--- | :--- | :---: | :---: | :--- |
| **Interactive Client Dashboard** | Next.js 16 (React 19, Tailwind) | **3000** | `http://localhost:3000` | Giao diện người dùng Web 4-Tab: Trích xuất, So sánh, Chat VQA, Lịch sử sổ cái |
| **Mobile Capture PWA** | Next.js Route `/mobile-capture` | **3000** | `http://localhost:3000/mobile-capture` | Giao diện chụp quét hóa đơn tối ưu cho điện thoại thông minh |
| **API Gateway & Business Logic** | FastAPI + Uvicorn (ASGI) | **8000** | `http://localhost:8000` | Điều phối xử lý EXIF, kiểm tra khớp số học $\Delta$, routing inference |
| **GPU Inference Worker** | PyTorch 2.5 + PEFT (Pop!_OS) | **8001** | `http://localhost:8001` | Chạy trực tiếp Qwen3-VL 4-bit NF4 + LoRA v2, hot-swap adapter, 4-step VRAM eviction |
| **Ollama Local Sandbox (Baseline)** | Ollama Linux Runtime | **11434**| `http://localhost:11434`| Chạy mô hình so sánh MiniCPM-V 2.6 (chế độ `keep_alive: 0`) |

---

## 2. CẤU HÌNH DATABASE & CONNECTION STRINGS

Hệ thống hỗ trợ cơ chế **Hybrid Database** (Ưu tiên đám mây Supabase PostgreSQL, tự động fallback về SQLite cục bộ nếu mất mạng):

### A. Cơ sở dữ liệu chính: Supabase PostgreSQL (Cloud)
* **Connection String (URI):**  
  `postgresql://postgres.qwprbxxxbvueozffqhdd:Hmd0410@2004@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres`
* **Host:** `aws-0-ap-southeast-1.pooler.supabase.com`
* **Port:** `5432`
* **Database Name:** `postgres`
* **User:** `postgres.qwprbxxxbvueozffqhdd`
* **Password:** `Hmd0410@2004`
* **SSL Mode:** `require`
* **Region:** `ap-southeast-1` (AWS Singapore)

### B. Cơ sở dữ liệu Fallback cục bộ: SQLite
* **Tập tin lưu trữ:** `backend/receipts_history.db`
* **Cơ chế:** Khi không có Internet hoặc không kết nối được PostgreSQL, backend tự động chuyển sang ghi và đọc từ `receipts_history.db` mà không làm gián đoạn dịch vụ của người dùng.

---

## 3. CẤU HÌNH DỊCH VỤ BÊN THỨ 3 (3RD-PARTY APIS & TOKENS)

Tất cả các biến môi trường cấu hình dịch vụ bên thứ 3 được lưu trữ an toàn trong file `backend/.env`:

```env
# 1. Supabase Cloud Storage & Auth Service
SUPABASE_URL=https://qwprbxxxbvueozffqhdd.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF3cHJieHh4YnZ1ZW96ZmZxaGRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4OTU5NzA4MywiZXhwIjoyMTA1MTczMDgzfQ.PwuArhKLDbeQrg_SLVSFxXqhoIUGJRDbawyVnayrooA

# 2. Database Connection Pooling
SUPABASE_DB_HOST=aws-0-ap-southeast-1.pooler.supabase.com
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=postgres.qwprbxxxbvueozffqhdd
SUPABASE_DB_PASSWORD=Hmd0410@2004
```

---

## 4. DANH SÁCH TÀI KHOẢN ĐĂNG NHẬP HỆ THỐNG DEMO

Hệ thống đã tích hợp sẵn tính năng **1-Click Demo Login** tại màn hình đăng nhập, hoặc người dùng có thể nhập trực tiếp thông tin sau:

| STT | Username / Email | Password | Họ và Tên | Đơn vị / Tổ chức | Vai trò (Role) | Quyền hạn trong hệ thống |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **1** | **`ketoan@winmart.vn`** | `123456` | Kế toán Trưởng WinMart | WinMart Retail Group | `accountant` | Toàn quyền kiểm toán, trích xuất hóa đơn, xuất file Excel/CSV, chỉnh sửa bảng hàng hóa |
| **2** | **`thungan@highlands.vn`** | `123456` | Thu ngân Ca 1 Highlands | Highlands Coffee VN | `cashier` | Quét hóa đơn camera, xem kết quả trích xuất, đối soát khớp tiền mặt ca trực |
| **3** | **`kiemtoan@fpt.edu.vn`** | `123456` | Kiểm toán viên Độc lập FPT | FPT Auditing & Analytics | `admin` | Quản trị toàn bộ lịch sử, xem telemetry GPU, kích hoạt dọn VRAM (`Kick VRAM`) |

---

## 5. HƯỚNG DẪN KHỞI CHẠY TOÀN BỘ HỆ THỐNG DEMO (1-CLICK)

### Cách 1: Chạy 1-Click trên Windows
Nhấp đúp chuột vào file:
👉 **`start_demo.bat`**  
*(Script sẽ tự động kiểm tra Python, cài đặt package, khởi động FastAPI backend port 8000, Next.js frontend port 3000 và tự động mở trình duyệt)*

### Cách 2: Chạy 1-Click trên Linux / macOS
Mở terminal tại thư mục gốc và chạy:
```bash
chmod +x start_demo.sh
./start_demo.sh
```

### Cách 3: Chạy thủ công từng thành phần
1. **Khởi chạy Backend:**
   ```bash
   pip install -r requirements.txt
   cd backend
   python -m uvicorn api:app --host 0.0.0.0 --port 8000
   ```
2. **Khởi chạy Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Mở trình duyệt truy cập: `http://localhost:3000`
