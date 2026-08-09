# BÁO CÁO TỔNG HỢP DATASET (KIE)
*Trạng thái CSDL phục vụ huấn luyện và đánh giá mô hình bóc tách thông tin hóa đơn tiếng Việt.*

---

## 1. PHÂN TÍCH CHI TIẾT 3 TẬP DỮ LIỆU LÕI

Để mô hình đạt được độ chính xác cao nhất (State-of-the-art) trên môi trường thực tế, đề tài không chỉ phụ thuộc vào một nguồn dữ liệu duy nhất mà kết hợp sức mạnh của 3 tập dữ liệu với các mục đích cụ thể:

### 1.1 Tập MC-OCR 2021 (Public Benchmark Dataset)
- **Nguồn gốc:** Trích xuất nguyên bản từ cuộc thi MC-OCR 2021 do VLSP (Cộng đồng Xử lý ngôn ngữ tự nhiên tiếng Việt) tổ chức. Nhóm quyết định giữ nguyên toàn bộ dữ liệu gốc để đảm bảo tính khách quan và công bằng khi so sánh Benchmark.
- **Số lượng:** **1.546 ảnh** (Bao gồm 1.155 ảnh tập Train và 391 ảnh tập Val/Test).
- **Đặc trưng:** Đây là tập dữ liệu có độ khó cực cao (Hard-examples). Các ảnh được thu thập từ cộng đồng (Crowdsourced) bằng nhiều loại thiết bị cũ, chứa nhiều nhiễu như: mất góc, nhàu nát, bóng râm che khuất, chữ in mờ đứt nét, ánh sáng yếu.
- **Vai trò:** Đóng vai trò là "Thước đo chuẩn". Mô hình sẽ được đem ra test trên tập này để so sánh độ F1-Score trực tiếp với các bài báo và giải pháp thương mại khác.

### 1.2 Tập Synthetic (Virtual Pre-training Dataset)
- **Nguồn gốc:** Dữ liệu nhân tạo (100% tự sinh) thông qua mã nguồn tự phát triển. Nhóm đã code tay 15 template HTML/CSS mô phỏng lại các thương hiệu bán lẻ lớn nhất Việt Nam. Dữ liệu văn bản (Sản phẩm, giá tiền, địa chỉ) được bơm ngẫu nhiên (Data Injection) thông qua Jinja2. Cuối cùng, dùng Playwright để render ra ảnh.
- **Tăng cường dữ liệu (Augmentation):** Không chỉ sinh ra ảnh phẳng (Flat images), nhóm còn áp dụng các thuật toán Computer Vision để lồng ghép hiệu ứng: chèn nền mặt bàn gỗ/nhựa/thảm, bẻ cong (Warping), đổ bóng (Shadowing) và làm nhiễu (Gaussian Noise).
- **Số lượng:** **10.000 ảnh** (Đã sinh thành công hoàn toàn tự động).
- **Vai trò:** Cung cấp nhãn Bounding Box với độ chính xác Pixel-perfect. Tập này dùng để "Dạy vỡ lòng" (Warm-up / Pre-train) cho mô hình LayoutLM, giúp nó hiểu được cấu trúc không gian (Spatial Layout) cơ bản của một tờ hóa đơn trước khi đụng tới dữ liệu thật.

### 1.3 Tập in ấn thực tế (Real-world Fine-tuning Dataset)
- **Nguồn gốc:** Nhóm tiến hành in hàng ngàn hóa đơn từ file PDF của tập Synthetic ra giấy thật (cả giấy in nhiệt siêu thị và giấy A4). Sau đó, thành viên nhóm sử dụng Camera điện thoại trực tiếp chụp lại dưới nhiều góc độ và điều kiện ánh sáng khác nhau để mô phỏng chính xác thói quen của người dùng cuối (End-user).
- **Labeling:** Ứng dụng quy trình Auto-labeling tối tân nhất: Chạy PaddleOCR lấy Box -> Nạp vào LLM (Gemini 1.5) để sửa lỗi chính tả và gán nhãn BIO tự động.
- **Vai trò:** Đây là "Vũ khí bí mật" của đồ án. Dùng để Fine-tune sâu mô hình, chuyển hóa kiến thức từ môi trường ảo (Synthetic) sang môi trường vật lý (Real-world).

---

## 2. TIẾN ĐỘ THU THẬP TẬP REAL-WORLD (SUPABASE)

Toàn bộ ảnh chụp thực tế đang được đồng bộ lên Supabase Storage và phân loại thành 2 nhóm: **Hóa đơn in nhiệt (raw_images)** và **Hóa đơn khổ lớn/Biên lai (large_invoices)**. Dưới đây là thống kê chi tiết tiến độ chụp và tải lên:

### ✅ ĐÃ HOÀN TẤT CHỤP VÀ TẢI LÊN (11/15 Templates)
*Trung bình ~130 ảnh/loại. Tổng số ảnh đã thu thập thành công: **1.441 ảnh**.*

**Nhóm siêu thị & Cửa hàng tiện lợi:**
- `supermarket_lotte` : 132 ảnh
- `supermarket_bachhoaxanh` : 133 ảnh
- `convenience_circlek` : 130 ảnh
- `convenience_7eleven` : 136 ảnh

**Nhóm F&B (Nhà hàng / Cafe):**
- `cafe_highlands` : 132 ảnh
- `cafe_phuclong` : 134 ảnh
- `cafe_starbucks` : 133 ảnh
- `restaurant_jollibee` : 132 ảnh

**Nhóm Hóa đơn khổ A4 / Biên lai (Bucket: large_invoices):**
- `einvoice_viettel` : 134 ảnh
- `einvoice_vnpt` : 134 ảnh
- `receipt_c45_bb` : 111 ảnh

### ⏳ ĐANG TIẾN HÀNH CHỤP VÀ BỔ SUNG (4/15 Templates)
*Team đang tiến hành in ấn và chụp bằng điện thoại. Sẽ sớm upload lên hệ thống.*

- `supermarket_winmart` : [ Đang cập nhật... ] ảnh
- `convenience_gs25` : [ Đang cập nhật... ] ảnh
- `restaurant_kfc` : [ Đang cập nhật... ] ảnh
- `minimart_anan` : [ Đang cập nhật... ] ảnh

---

## 3. KẾT LUẬN & ĐỀ XUẤT
* **Mục tiêu Data:** Ngay khi hoàn tất chụp 4 template còn lại, tổng kích thước tập Real-world sẽ cán mốc **~2.000 ảnh**, tạo thành bộ 3 Dataset hoàn hảo: **1.5K (MC-OCR) - 10K (Synthetic) - 2K (Real-world)**.
* **Mục tiêu Model:** Tiến hành trích xuất tọa độ PaddleOCR và dùng Gemini API để tự động làm sạch (Clean) & gán nhãn toàn bộ 2.000 ảnh thực tế. Sau đó nạp vào pipeline huấn luyện LayoutLM/PhoBERT.
