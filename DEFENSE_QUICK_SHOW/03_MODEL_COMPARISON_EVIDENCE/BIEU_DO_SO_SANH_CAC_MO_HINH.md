# 📊 BẢNG ĐỐI ĐẦU SO SÁNH CÁC MÔ HÌNH (HEAD-TO-HEAD BENCHMARK EVIDENCE)

## 1. Bảng Hiệu Năng Tổng Thể Trên Tập Test VietInvoice (1,166 Hóa Đơn)

| STT | Kiến trúc mô hình | Cơ chế xử lý | Macro-F1 (%) | Exact Match (%) | Item Recall (%) | Item Precision (%) | Tỉ lệ lỗi số học (%) | Latency (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Qwen3-VL 8B (LoRA v2)** *(Đề xuất chính)* | **End-to-End Multimodal VLM (Prompt v2)** | **93.35%** | **86.87%** | **98.63%** | **97.12%** | **1.2%** | **8.25s** |
| 🥈 **2** | **Two-Stage Hybrid** *(Nghiên cứu đối sánh)* | **PaddleOCR + Spatial Grouping + Qwen2.5** | **89.20%** | **84.50%** | **96.80%** | **95.40%** | **0.8%** | **7.50s** |
| 🥉 **3** | **Qwen3-VL 8B (LoRA v1)** | End-to-End VLM (Prompt v1 - Flat Array) | 88.62% | 85.12% | 94.20% | 93.80% | 4.8% | 8.10s |
| 4 | **Qwen3-VL 8B (Base Zero-Shot)** | VLM nguyên bản không qua tinh chỉnh | 86.85% | 87.50% | 96.13% | 94.50% | 6.5% | 14.20s |
| 5 | **LayoutLMv3 (Multimodal Transformer)** | Pipeline 2 tầng: PaddleOCR + LayoutLMv3 | 84.10% | 76.20% | 88.50% | 87.10% | 11.4% | 3.20s |
| 6 | **PhoBERT-base (NER Pipeline)** | Pipeline 2 tầng: PaddleOCR + PhoBERT | 74.20% | 62.40% | 78.90% | 76.50% | 18.7% | 2.80s |
| 7 | **DeepSeek-OCR + Qwen2.5 (7B)** | Pipeline 2 tầng: DeepSeek + LLM | 66.93% | 53.64% | 86.38% | 84.10% | 14.2% | 12.57s |
| 8 | **DeepSeek-OCR + Heuristic Regex** | Pipeline 2 tầng truyền thống: OCR + Regex | 52.42% | 52.89% | 32.42% | 41.20% | 38.6% | 7.78s |
| 9 | **MiniCPM-V 2.6 (8B Zero-Shot)** | VLM mã nguồn mở đối sánh | 42.92% | 23.73% | 54.22% | 51.10% | 42.1% | 15.60s |

---

## 2. Bảng Đánh Giá Khả Năng Tổng Quát Hóa Ngoài Miền (499 Ảnh MCOCR Thực Tế)

| Kiến trúc mô hình | Macro-F1 (%) | Exact Match (%) | Line Item Recall (%) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen3-VL 8B (LoRA v2)** | **74.10%** | **68.40%** | **88.30%** | **Vượt trội trên ảnh thực tế mờ, nhăn, nghiêng góc** |
| **LayoutLMv3 (Fine-tuned)** | 61.20% | 52.10% | 71.40% | Giảm sút mạnh do lỗi OCR lan truyền (Cascading Errors) |
| **PhoBERT-base (Fine-tuned)** | 53.80% | 44.50% | 62.10% | Mất hoàn toàn cấu trúc không gian 2 chiều |
| **DeepSeek + Regex** | 31.40% | 28.50% | 24.60% | Thất bại hoàn toàn trên bố cục hóa đơn không tiêu chuẩn |

---

## 3. Phân Tích So Sánh Chuyên Sâu: Pure VLM vs. Two-Stage Hybrid

Khi hội đồng hỏi: *"Tại sao nhóm lại đề xuất Pure VLM thay vì Two-Stage Hybrid có Bounding Box?"*

| Tiêu chí | Pure End-to-End VLM (`Qwen3-VL LoRA v2`) | Two-Stage Hybrid (`PaddleOCR + Spatial + Qwen2.5`) |
| :--- | :--- | :--- |
| **Kiến trúc** | **1 mô hình duy nhất** giải quyết tất cả từ Pixel $\rightarrow$ JSON. | **3 thành phần rời rạc:** OCR Detector + Spatial Merger + LLM. |
| **Hiện tượng lỗi lan truyền (Cascading)** | **Không có**. Đọc trực tiếp từ ảnh gốc, chữ mờ vẫn phán đoán ngữ cảnh tốt. | **Có rủi ro cao**. Nếu PaddleOCR đọc sót hoặc sai chữ, các tầng sau không thể sửa. |
| **Xử lý chữ nghệ thuật / logo** | **Rất mạnh**. Vision Encoder nhận diện được cả logo Highlands, WinMart, font stylized. | **Yếu hơn**. OCR detector thường bỏ sót logo hoặc font cách điệu. |
| **Xử lý bảng biểu rớt dòng (N-to-M)** | Có nguy cơ lệch dòng nếu ảnh nghiêng $\implies$ Đã khắc phục bằng **Price Cascading Solver**. | **Khắc phục triệt để bằng tọa độ Bounding Box vật lý** trên trục $Y$. |
| **Độ phức tạp triển khai (Deployment)** | **Rất thấp**. Chỉ cần nạp 1 mô hình vào GPU. | **Phức tạp**. Cần quản lý đồng thời cả môi trường PaddleOCR và Ollama/LLM. |
