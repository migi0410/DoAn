# BẢNG TỔNG HỢP TOÀN BỘ SỐ LIỆU THỰC NGHIỆM ĐÃ ĐƯỢC XÁC THỰC (OFFICIAL AUDITED NUMBERS)
## Đề tài: AVIR-KIE (AIP491 - Capstone Project, Đại học FPT)
*Mọi con số dưới đây đều được trích xuất và tính toán trực tiếp từ tệp nhật ký thực nghiệm `official_benchmark/benchmark_results.csv` (9,328 lượt suy luận), tập dữ liệu `FINAL_SPLIT_JSONL_ONLY`, và nhật ký huấn luyện GPU RTX 5060 Ti.*

---

## 1. THỐNG KÊ BỘ DỮ LIỆU VIETINVOICE & CÁC TẬP PHÂN CHIA (DATASET SPLITS)

Tổng quy mô kho dữ liệu sử dụng: **12,799 hình ảnh hóa đơn**.

| Tập dữ liệu (Split) | Tổng số mẫu | Nguồn VietInvoice-Syn (Sinh tự động) | Nguồn VietInvoice-Real (Ảnh chụp thực tế) | Nguồn MC-OCR (Benchmark công khai) | Tỷ lệ phân chia |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** (Huấn luyện) | **9,322** | 7,987 | 1,335 | — | 80.0% (In-domain) |
| **Validation Set** (Đánh giá) | **1,165** | 990 | 175 | — | 10.0% (In-domain) |
| **Test Set** (Kiểm thử độc lập) | **1,166** | 998 | 168 | — | 10.0% (In-domain) |
| **MC-OCR Test Set** (Out-of-domain) | **1,146** *(subset chọn lọc 499)* | — | — | 1,146 | Kiểm thử ngoại suy |
| **TỔNG CỘNG** | **12,799** | **9,975** | **1,678** | **1,146** | **100.0%** |

*Ghi chú:*
- Tập `train.jsonl` ban đầu trước khi chia 80/10/10 là: **11,653 mẫu** ($9,322 + 1,165 + 1,166 = 11,653$).
- Số lượng template thương hiệu thực tế: **14 chuỗi bán lẻ** (WinMart, Circle K, Highlands Coffee, Phúc Long, GS25, Bách Hóa Xanh, Lotte Mart, v.v.).

---

## 2. THÔNG SỐ CẤU HÌNH HUẤN LUYỆN 4-BIT QLORA (TRAINING HYPERPARAMETERS)

| Tham số cấu hình | Giá trị cấu hình chuẩn | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Base Foundation Model** | `Qwen/Qwen3-VL-8B-Instruct` | 8.32 tỷ tham số tổng |
| **Kiểu Lượng tử hóa (Quantization)** | 4-bit NormalFloat (NF4) | Double Quantization enabled, compute dtype = `bfloat16` |
| **LoRA Rank ($r$)** | **32** | Tăng dung lượng biểu diễn cấu trúc bảng |
| **LoRA Alpha ($\alpha$)** | **64** | Tỷ lệ scale $\alpha / r = 2.0$ |
| **LoRA Dropout** | **0.05** | Chống overfitting trên tập train |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Bao phủ toàn bộ Self-Attention và MLP layers |
| **Số tham số huấn luyện** | **67.1 Triệu** | Chiếm **0.80%** tổng trọng số mô hình |
| **Batch Size hiệu dụng** | **8** | Per-device batch size = 2, Gradient Accumulation = 4 |
| **Tốc độ học (Learning Rate)** | **1.0e-4** | Cosine Annealing decay, 50 steps Warmup |
| **Số bước tối ưu (Steps)** | **1,166 steps** | Tương đương 1 epoch trên 9,322 mẫu train |
| **Thời gian huấn luyện** | **4.4 giờ** | Thực thi trên 1 GPU NVIDIA RTX 5060 Ti 16GB VRAM |
| **Loss hội tụ cuối cùng** | **0.0115** | Giảm mượt mà từ mức ban đầu 2.45 |

---

## 3. BẢNG HIỆU NĂNG TỔNG THỂ TRÊN 1,166 HÓA ĐƠN TEST HELD-OUT (TABLE VI.1)

*Kết quả trung bình cộng trên toàn bộ 1,166 mẫu kiểm thử in-domain độc lập:*

| STT | Kiến trúc mô hình | Trường phái / Schema | Macro-F1 (%) | Exact Match (%) | Line Item Recall (%) | Line Item Prec (%) | Hợp lệ JSON (%) | Độ trễ (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 👑 **Qwen3-VL (8B) - LoRA v2 (Enhanced)** | **End-to-End VLM / Prompt v2** | **93.35%** | **95.09%** | **98.63%** | **98.64%** | **99.9%** | **8.25s** |
| **2** | 🥈 **Qwen3-VL (8B) - LoRA v1 (Ablation)** | End-to-End VLM / Prompt v1 | **92.37%** | **93.20%** | **93.50%** | **93.51%** | **100.0%** | **7.12s** |
| **3** | 🥉 **Qwen3-VL (8B) - Base (Prompt v2)** | Foundation Zero-Shot / Prompt v2 | **86.89%** | **88.14%** | **97.94%** | **98.12%** | **100.0%** | **22.72s** |
| **4** | 🔹 **Qwen3-VL (8B) - Base (Prompt v1)** | Foundation Zero-Shot / Prompt v1 | **86.85%** | **87.50%** | **96.13%** | **96.30%** | **100.0%** | **14.20s** |
| **5** | 🧠 **DeepSeek-OCR + Qwen2.5 (7B)** | Pipeline 2 tầng OCR + LLM | **66.93%** | **53.64%** | **86.38%** | **86.87%** | **97.7%** | **12.57s** |
| **6** | ⚡ **DeepSeek-OCR + Heuristic Regex** | Pipeline 2 tầng OCR + Regex | **52.42%** | **52.89%** | **32.42%** | **30.95%** | **100.0%** | **7.78s** |
| **7** | 🐢 **MiniCPM-V 2.6 (8B)** | Zero-Shot Multimodal VLM | **42.92%** | **23.74%** | **54.22%** | **57.63%** | **99.2%** | **15.60s** |

---

## 4. CHI TIẾT F1-SCORE TRÊN TỪNG TRƯỜNG THỰC THỂ (8 KIE FIELDS) (%)

| Mô hình | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | ITEM_NAME | ITEM_QTY | ITEM_PRICE | ITEM_AMOUNT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **Qwen3-VL LoRA v2 (Enhanced)** | **99.40%** | **97.86%** | **99.31%** | **98.54%** | **98.63%** | **85.96%** | **68.99%** | **98.07%** |
| 🥈 **Qwen3-VL LoRA v1 (Prompt v1)** | 99.23% | 92.20% | 98.89% | 98.20% | 93.50% | 86.25% | 73.82% | 96.91% |
| 🥉 **Qwen3-VL Base (Prompt v2)** | 98.80% | 91.25% | 97.17% | 98.97% | 98.01% | 67.53% | 55.04% | 88.36% |
| 🔹 **Qwen3-VL Base (Prompt v1)** | 98.97% | 91.51% | 95.97% | 96.31% | 96.20% | 66.10% | 64.87% | 84.85% |
| 🧠 **DeepSeek + Qwen2.5** | 67.32% | 81.22% | 59.01% | 76.67% | 86.45% | 55.48% | 48.35% | 60.90% |
| ⚡ **DeepSeek + Regex** | 56.86% | 82.68% | 84.39% | 64.58% | 31.36% | 32.46% | 37.62% | 29.43% |
| 🐢 **MiniCPM-V 2.6** | 50.09% | 49.23% | 71.18% | 55.06% | 55.01% | 18.35% | 15.27% | 29.16% |

---

## 5. THỰC NGHIỆM NGOẠI SUY ZERO-SHOT TRÊN TẬP MC-OCR THỰC TẾ (499 MẪU) (TABLE VI.2)

| Cấu hình mô hình | Prompt Template | Hợp lệ JSON (%) | Seller Sim (%) | Total Sim (%) | Item Recall (%) | Macro-F1 (%) | Độ trễ (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **Qwen3-VL LoRA v2 (Enhanced)** | Enhanced Prompt v2 | **98.4%** | **74.8%** | **74.7%** | **88.3%** | **74.1%** | **7.68s** |
| 🥈 **Qwen3-VL LoRA v2 (Initial)** | Initial Prompt v2 | 97.2% | 73.7% | 74.5% | 86.0% | 71.0% | 7.42s |

---

## 6. MA TRẬN ABLATION FACTORIAL 2x2 (TABLE VI.3)

*Tác động tương hỗ giữa Kỹ thuật Tinh chỉnh Tham số (Fine-Tuning) và Thiết kế Cấu trúc Schema (Prompt Engineering):*

| Phương thức huấn luyện | Định dạng Schema đầu ra | Macro-F1 (%) | Line Item Recall (%) | Tốc độ suy luận (s) | Đánh giá & Nhận xét |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Zero-Shot Base Foundation** | Prompt v1 (Flat Parallel Arrays) | 86.85% | 96.13% | 14.20s | Mẫu gốc chưa tinh chỉnh, output dạng mảng rời rạc |
| **Zero-Shot Base Foundation** | Prompt v2 (Hierarchical Schema) | 86.89% *(+0.04%)* | 97.94% *(+1.81%)* | 22.72s *(+60%)* | Mô hình sinh JSON lồng nhau chi tiết hơn nên tốn thời gian hơn |
| **4-bit QLoRA Fine-Tuned** | Prompt v1 (Flat Parallel Arrays) | 92.37% *(+5.52%)* | 93.50% | 7.12s *(-49.8%)* | Tăng tốc mạnh nhờ LoRA học được điểm dừng sớm |
| **4-bit QLoRA Fine-Tuned (Proposed)** | Prompt v2 Enhanced (Multi-Line Grouping) | **93.35%** *(+6.50%)* | **98.63%** *(+2.50%)* | **8.25s** *(-63.7%)* | **Tối ưu toàn diện**: Vừa giải quyết lệch dòng, vừa tăng tốc và đạt F1 cao nhất |

---

## 7. CHỈ SỐ HẠ TẦNG KỸ THUẬT & QUẢN LÝ BỘ NHỚ VRAM (ENGINEERING METRICS)

| Trạng thái hệ thống | VRAM Allocated | VRAM Reserved | Thời gian chuyển đổi (Switching Latency) | Cơ chế thực thi |
| :--- | :---: | :---: | :---: | :--- |
| **Base Model + LoRA v2 Resident** | **6,215.9 MB** | 6,330.0 MB | — | PyTorch 4-bit NF4 GPU Serving |
| **Chuyển đổi LoRA v2 $\rightarrow$ Base** | 6,215.9 MB | 6,330.0 MB | **0.00 ms** (Tức thì) | `with model.disable_adapter():` |
| **Chuyển đổi Base $\rightarrow$ LoRA v1** | 6,215.9 MB | 6,330.0 MB | **0.00 ms** (Tức thì) | `model.set_adapter("v1")` |
| **Sau lệnh Kick VRAM (`/api/gpu/unload`)** | **9.1 MB** | **30.0 MB** | **0.42 s** | `cpu -> del -> gc -> empty_cache -> ipc_collect` (Xả sạch 99.9%) |
| **Auto Lazy-Reload khi có request mới** | 6,223.2 MB | 6,612.0 MB | **34.70 s** | Tự động nạp lại trọng số 4-bit không cần restart server |
| **Chạy Ollama MiniCPM-V 2.6 (8B)** | ~5,500.0 MB (Ollama) | — | 15.52 s | `keep_alive: 0` tự động giải phóng VRAM ngay sau khi sinh |
| **Chạy song song Qwen3 + MiniCPM-V** | 11,725.0 MB (Đỉnh) | < 13,000 MB | 22.08 s | Không bao giờ tràn 16GB VRAM của RTX 5060 Ti |
