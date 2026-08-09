# MÔ TẢ CHI TIẾT TẬP DỮ LIỆU (DATASET DESCRIPTION)
*Tài liệu phục vụ Giảng viên Hướng dẫn (GVHD) và lên Outline bài báo khoa học (Paper).*

---

## 1. TỔNG QUAN VỀ TẬP DỮ LIỆU ĐỀ TÀI
Để giải quyết bài toán Trích xuất Thông tin Từ khóa (Key Information Extraction - KIE) trên hóa đơn tiếng Việt, đề tài sử dụng cách tiếp cận **kết hợp dữ liệu tổng hợp (Synthetic Data) và dữ liệu thực tế (Real-world Data)**. 

Toàn bộ Dataset được chia thành 3 tập con (Sub-datasets) với các vai trò riêng biệt trong quá trình Huấn luyện (Training), Tinh chỉnh (Fine-tuning) và Đánh giá (Evaluation):

1. **MC-OCR Subset:** Tập benchmark công khai.
2. **VietInvoice-Syn (10K):** Tập dữ liệu hóa đơn tổng hợp (Sinh tự động).
3. **VietInvoice-Real (~1.5K):** Tập dữ liệu hóa đơn in ấn thực tế.

---

## 2. CHI TIẾT CÁC TẬP DỮ LIỆU (SUB-DATASETS)

### 2.1 Tập MC-OCR (MC-OCR 2021 KIE Subset)
* **Số lượng:** 1.546 ảnh (Bao gồm 1.155 ảnh tập Train và 391 ảnh tập Val/Test).
* **Nguồn gốc:** Trích xuất nguyên bản từ tập dữ liệu gốc của cuộc thi khoa học dữ liệu VLSP MC-OCR 2021 (Task 3: Key Information Extraction). Nhóm giữ nguyên toàn bộ dữ liệu gốc để đảm bảo tính khách quan khi so sánh.
* **Đặc điểm:** Hóa đơn đa phần là ảnh chụp thực tế từ năm 2020 trở về trước, chất lượng ảnh thay đổi đa dạng (mờ, nhòe, nhăn nheo, bóng râm).
* **Vai trò trong Paper:** 
  - Đóng vai trò là tập **Public Benchmark**. 
  - Dùng để so sánh hiệu năng (F1-Score, Precision, Recall) của mô hình đề xuất với các baseline models có sẵn trên thị trường.

### 2.2 Tập VietInvoice-Syn (Synthetic Dataset)
* **Số lượng:** 10.000 ảnh.
* **Quy trình xây dựng (Data Pipeline):**
  - Khảo sát và xây dựng **15 Templates HTML/CSS** dựa trên các thương hiệu phổ biến tại Việt Nam (Siêu thị Lotte, BHX, Cafe Highlands, Hóa đơn điện tử Viettel...).
  - Phát triển script Python (Jinja2 + Playwright) đâm dữ liệu ngẫu nhiên (sản phẩm, giá, địa chỉ, timestamp) vào các template và render ra ảnh.
  - Áp dụng các kỹ thuật Tăng cường dữ liệu không gian (Spatial Augmentation): Ghép nền mặt bàn (gỗ, thảm), đổ bóng (Shadowing), tạo nếp gấp (Fold effects) để mô phỏng hóa đơn thật.
* **Ưu điểm:** Cung cấp nhãn (Ground Truth) Bounding Box hoàn hảo đến từng Pixel (Pixel-perfect), loại bỏ hoàn toàn sai số của con người (Human-error) trong quá trình Labeling.
* **Vai trò trong Paper:** 
  - Đóng vai trò là tập **Pre-training / Warm-up**. 
  - Giúp các mô hình ngôn ngữ không gian (như LayoutLM) học được cấu trúc (Layout) và vị trí tương đối của 8 trường thông tin (SELLER, ADDRESS, TOTAL_COST...) trước khi fine-tune trên dữ liệu thật.

### 2.3 Tập VietInvoice-Real (Real-world Printed Dataset)
* **Số lượng hiện tại:** 1.441 ảnh (Dự kiến hoàn thiện: 2.000 ảnh).
* **Quy trình xây dựng:** 
  - In ấn các hóa đơn được sinh từ tập Synthetic ra giấy thật (bao gồm cả giấy in nhiệt bill nhỏ và giấy A4 cho hóa đơn điện tử).
  - Sử dụng Camera điện thoại để chụp thủ công dưới nhiều góc độ (Nghiêng, xoay), điều kiện ánh sáng (Sáng, tối, chói) và môi trường (Cầm tay, để trên bàn).
  - Đưa lên hệ thống Supabase Storage và sử dụng quy trình **Hybrid Labeling (PaddleOCR + Gemini 1.5 API)** để tự động hóa việc gán nhãn BIO và sửa lỗi chính tả OCR (Error Correction).
* **Thống kê Phân bổ hiện tại (Storage Buckets):**
  - **Bucket `raw_images` (Bill in nhiệt):** 1.062 ảnh (Bao gồm Lotte, BHX, CircleK, 7Eleven, Highlands, Phúc Long, Starbucks, Jollibee).
  - **Bucket `large_invoices` (Bill A4/Biên lai):** 379 ảnh (Bao gồm Hóa đơn Viettel [134], VNPT [134], và Biên lai C45 [111]).
  - **Đang thiếu:** Winmart, GS25, KFC, Minimart.
* **Vai trò trong Paper:** 
  - Đóng vai trò là tập **Fine-tuning và Test Set chính**. 
  - Thể hiện sự đóng góp lớn nhất của đề tài (Contributions): Xây dựng thành công quy trình khép kín từ sinh dữ liệu ảo (Virtual) -> in ấn (Physical) -> gán nhãn tự động (Auto-labeling) bằng AI, giải quyết bài toán thiếu hụt dữ liệu KIE tiếng Việt.

---

## 3. CẤU TRÚC NHÃN (LABEL SCHEMA)
Cả 3 tập dữ liệu trên đều được đồng nhất hóa quy chuẩn gán nhãn theo 8 trường thông tin lõi (Core Fields), bao gồm 4 trường Header và 4 trường Line-item:
1. `SELLER`: Tên đơn vị bán hàng.
2. `ADDRESS`: Địa chỉ phát hành hóa đơn.
3. `TIMESTAMP`: Dấu thời gian (Ngày/Giờ).
4. `TOTAL_COST`: Tổng số tiền thanh toán.
5. `ITEM_NAME`: Tên món hàng/sản phẩm.
6. `ITEM_QTY`: Số lượng mua.
7. `ITEM_PRICE`: Đơn giá sản phẩm.
8. `ITEM_AMOUNT`: Thành tiền của sản phẩm đó.
