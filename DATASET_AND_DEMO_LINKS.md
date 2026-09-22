# KHO LƯU TRỮ TÀI NGUYÊN ĐỒ ÁN AVIR-KIE (DATASET, MODEL CHECKPOINTS & DEMO)
### Dự án: AVIR-KIE (Mã đề tài: SU26AI50 - Nhóm: GSU26AI01)
**Hội đồng chấm bảo vệ:** Đồ án Tốt nghiệp Kỹ sư Trí tuệ Nhân tạo (AIP491) – FPT University TP.HCM  
**Giảng viên hướng dẫn:** Thầy Nguyễn Hồng Hải (`hainh51@fe.edu.vn`)

---

## 🔗 LIÊN KẾT GOOGLE DRIVE DUY NHẤT CỦA ĐỒ ÁN (OFFICIAL REPOSITORY)
Toàn bộ tài nguyên kích thước lớn (Dataset 12K ảnh, Model Weights, Video Demo) được lưu trữ tập trung tại **1 liên kết Google Drive duy nhất**:

👉 **`https://drive.google.com/drive/folders/1JQb53Qe1KcIs2eO8WF1Vj9B2Fd-vwMsp?usp=drive_link`**  
*(Nhóm đã cấp quyền truy cập công khai cho Giảng viên hướng dẫn và Hội đồng thẩm định FPT)*

---

## 📂 CẤU TRÚC THƯ MỤC BÊN TRONG GOOGLE DRIVE

Thư mục Google Drive trên bao gồm 3 phân khu chính:

### 1. Thư mục `01_Dataset/`
* **`VietInvoice_Annotations_and_Splits.zip`** (~70 MB):
  * `OFFICIAL_DATASET/`: `train.jsonl` (9,322 mẫu), `val.jsonl` (1,165 mẫu), `test.jsonl` (1,166 mẫu).
  * `test_500.jsonl` / `test_mcocr_official.jsonl`: 500 ảnh hóa đơn thực tế MC-OCR 2021 dùng cho Out-of-domain Generalization Benchmark.
  * `OFFICIAL_LAYOUTLM_DATASET_v2/`: Bộ dữ liệu định dạng Hugging Face Dataset cho baseline LayoutLMv3.
  * `PhoBERT_NER_Data/`: `test_official_bert.json`, `test_500_bert.json`.
  * `FINAL_BBOX_DATASET_V3.json`: Tọa độ Bounding Box cấp từ (word-level) cho các bài toán OCR/KIE truyền thống.
* **`VietInvoice_Full_Images_12K.zip`** (~5.8 GB):
  * Trọn bộ 12,046 ảnh hóa đơn bán lẻ tiếng Việt (Synthetic Playwright + Chụp thực tế + MC-OCR).

### 2. Thư mục `02_Model_Checkpoints/`
* **`Qwen3_VL_8B_LoRA_Checkpoint.zip`** (~61.4 MB):
  * `adapter_model.safetensors`
  * `adapter_config.json`
  * Dùng trực tiếp nạp vào Qwen2/3-VL-8B để suy luận KIE tiếng Việt với Macro-F1 93.35%.

### 3. Thư mục `03_Demo_Video/`
* **`Product_Demo_Video.mp4`** (~62.38 MB):
  * Video quay màn hình chi tiết 4 tab chức năng của ứng dụng web AVIR-KIE.
  * *(File này cũng đã được tích hợp sẵn trong gói nộp Edunext `Source_Code_and_Demo_Video.zip`)*.

---

## 💡 DỮ LIỆU MẪU ĐI KÈM TRONG MÃ NGUỒN (TEST NHANH KHÔNG CẦN TẢI DRIVE)
Trong gói mã nguồn `Source_Code_and_Database.zip` nộp trên hệ thống, thư mục `backend/templates_images/` đã có sẵn 5 mẫu hóa đơn đại diện tiêu biểu (WinMart, Highlands Coffee, Circle K, Phúc Long, Viettel e-Invoice). Hội đồng có thể trải nghiệm suy luận và đối soát ngay lập tức mà không cần tải dữ liệu lớn từ Google Drive.
