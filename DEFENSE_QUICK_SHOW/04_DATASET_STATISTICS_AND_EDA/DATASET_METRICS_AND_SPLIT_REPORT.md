# 📈 BÁO CÁO THỐNG KÊ TẬP DỮ LIỆU & PHÂN BỐ NHÃN (DATASET METRICS & EDA)

## 1. Phân Chia Tập Dữ Liệu (Dataset Splits)

Hệ thống AVIR-KIE được xây dựng và kiểm định trên tổng cộng **13,690 hóa đơn**:

| Tập dữ liệu | Số lượng mẫu | Tỉ lệ (%) | Nguồn dữ liệu | Mục đích sử dụng |
| :--- | :---: | :---: | :--- | :--- |
| **Training Set** | **10,000** | 73.0% | VietInvoice Synthetic + Real Templates | Huấn luyện mô hình Qwen3-VL 8B QLoRA v2 |
| **Validation Set** | **2,025** | 14.8% | Held-out Synthetic Hardcore | Tinh chỉnh siêu tham số và chọn best checkpoint |
| **Test Set (In-Domain)** | **1,166** | 8.5% | VietInvoice Test Benchmark | Đánh giá chính thức độ chính xác (Macro-F1 93.35%) |
| **Test Set (Out-of-Domain)** | **499** | 3.7% | MCOCR Real Physical Receipts | Đánh giá khả năng tổng quát hóa trên ảnh thực tế mờ/nhăn/nghiêng |
| **TỔNG CỘNG** | **13,690** | **100.0%** | Đa dạng 12 lĩnh vực bán lẻ | Toàn bộ hệ thống |

---

## 2. Các Thực Thể Thông Tin Trích Xuất (Target Entities)

| Trường thông tin | Kiểu dữ liệu | Tỉ lệ xuất hiện trong tập dữ liệu | Mô tả chi tiết |
| :--- | :--- | :---: | :--- |
| `SELLER` | Chuỗi ký tự (String) | 99.8% | Tên thương hiệu, chuỗi bán lẻ hoặc cửa hàng xuất hóa đơn |
| `ADDRESS` | Chuỗi ký tự (String) | 96.4% | Địa chỉ chi nhánh, số nhà, quận/huyện, tỉnh/thành phố |
| `TIMESTAMP` | Chuỗi chuẩn hóa (DD/MM/YYYY) | 97.2% | Ngày và giờ thực hiện giao dịch thanh toán |
| `ITEMS` | Danh sách đối tượng (Array of Dicts) | 99.9% | Danh sách chi tiết từng món hàng trong giao dịch: |
| ↳ `name` | Chuỗi ký tự | 100% trong ITEMS | Tên món hàng, sản phẩm, quy cách kích cỡ |
| ↳ `qty` | Số nguyên/thực (String/Float) | 98.2% trong ITEMS | Số lượng món hàng đã mua |
| ↳ `price` | Chuỗi số tiền (Currency) | 89.5% trong ITEMS | Đơn giá niêm yết của 1 đơn vị sản phẩm |
| ↳ `amount` | Chuỗi số tiền (Currency) | 99.7% trong ITEMS | Thành tiền của món hàng sau khi nhân số lượng |
| `TOTAL_COST` | Chuỗi số tiền (Currency) | 100.0% | Tổng số tiền thanh toán thực tế cuối cùng của hóa đơn |

---

## 3. Các Biểu Đồ Minh Chứng Kèm Theo Trong Thư Mục Này
1. `eda_labels.png`: Biểu đồ tần suất xuất hiện và phân bố các nhãn thực thể KIE.
2. `eda_lengths.png`: Biểu đồ phân bố độ dài chuỗi ký tự và số lượng từ trong hóa đơn.
3. `sample_annotation_prompt_v2.json`: File mẫu minh họa cấu trúc dữ liệu JSONL chuẩn đầu vào của Qwen3-VL.
