# 📊 HƯỚNG DẪN KIỂM TRA DỮ LIỆU HUẤN LUYỆN & KIỂM THỬ (DATASET & ANNOTATIONS)

> **Mục đích:** Thư mục này chứa toàn bộ các file nhãn định dạng JSONL, số liệu thống kê EDA, và các ảnh mẫu thực tế đại diện cho các thử thách khó nhất trong trích xuất thông tin hóa đơn tiếng Việt.

---

## 📁 DANH MỤC CÁC FILE TRONG THƯ MỤC NÀY

### 1. File Nhãn JSONL (Ground Truth)
| Tên File | Số lượng mẫu | Mô tả |
| :--- | :--- | :--- |
| `vietinvoice_test_1166.jsonl` | **1,166** | Tập kiểm thử chính thức VietInvoice (dùng để tính điểm F1 93.35%, Line Item Recall 98.63%). |
| `mcocr_test_500.jsonl` | **500** | Tập kiểm thử hóa đơn thực tế MCOCR chụp ngoài đời thực (dùng để tính điểm F1 74.10% out-of-domain). |
| `vietinvoice_val_1165.jsonl` | **1,165** | Tập xác thực (Validation Set) dùng trong quá trình huấn luyện để chọn checkpoint tốt nhất. |
| `vietinvoice_train_sample_100.jsonl` | **100** | File mẫu 100 dòng đầu tiên của tập Train. **Khuyên dùng mở file này bằng Notepad/VS Code** để hiển thị tức thì, không bị lag. |
| `vietinvoice_train_full_9322.jsonl` | **9,322** | Toàn bộ tập huấn luyện đầy đủ (Train Set) đã được làm sạch và chuẩn hóa. |

---

### 2. Thư Mục Ảnh Mẫu Kiểm Thử Thực Tế (`sample_images/`)
* **`sample_images/vietinvoice_samples/`**: 10 ảnh hóa đơn đại diện từ tập Test VietInvoice (siêu thị, cửa hàng tiện lợi, nhà hàng, quán cà phê với định dạng rõ nét, hóa đơn dài nhiều món, bố cục dạng bảng).
* **`sample_images/mcocr_samples/`**: 10 ảnh hóa đơn chụp thực tế ngoài đời từ tập MCOCR (hóa đơn bị gấp nếp, giấy nhiệt phai mực, chụp nghiêng góc, ánh sáng yếu, bóng mờ).

---

### 3. Biểu Đồ Phân Bố Thống Kê (EDA)
* **`eda_labels.png`**: Biểu đồ cột thể hiện phân bố tần suất xuất hiện của các nhãn (`SELLER`, `ADDRESS`, `TIMESTAMP`, `TOTAL_AMOUNT`, `TAX_AMOUNT`, `ITEMS`).
* **`eda_lengths.png`**: Biểu đồ phân bố độ dài chuỗi ký tự và số lượng token cho từng trường dữ liệu.
* **`sample_annotation_prompt_v2.json`**: File JSON mẫu thể hiện trực quan cấu trúc một bản ghi hội thoại chuẩn đầu vào của Qwen3-VL theo Prompt v2.

---

## 📝 CẤU TRÚC ĐỊNH DẠNG MỘT MẪU JSONL (PROMPT V2)

Mỗi dòng trong file JSONL là một đối tượng JSON độc lập, tuân thủ định dạng hội thoại chuẩn của mô hình Vision-Language:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nTrích xuất thông tin hóa đơn tiếng Việt theo định dạng JSON với cấu trúc phân cấp..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"SELLER\": \"CÔNG TY TNHH HIGHLANDS COFFEE\",\n  \"ADDRESS\": \"Tầng 1, TTTM Vincom Center, Q.1, TP.HCM\",\n  \"TIMESTAMP\": \"2026-09-26 14:30:00\",\n  \"TOTAL_AMOUNT\": \"232.000\",\n  \"ITEMS\": [\n    {\n      \"name\": \"Tra Sen Vang S - ADJ\",\n      \"quantity\": \"1\",\n      \"price\": \"49.000\",\n      \"total\": \"49.000\"\n    },\n    {\n      \"name\": \"Com va Kem La Dua L\",\n      \"quantity\": \"1\",\n      \"price\": \"55.000\",\n      \"total\": \"55.000\"\n    }\n  ]\n}"
    }
  ],
  "images": ["path/to/invoice_image.jpg"]
}
```
