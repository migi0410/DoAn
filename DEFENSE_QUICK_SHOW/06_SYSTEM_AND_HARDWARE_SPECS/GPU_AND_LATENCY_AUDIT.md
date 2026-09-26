# 🖥️ BÁO CÁO KIỂM TOÁN TÀI NGUYÊN PHẦN CỨNG & ĐỘ TRỄ (HARDWARE SPECS & LATENCY AUDIT)

## 1. Cấu Hình Phần Cứng Hệ Thống

| Thành phần | Máy chủ Cục bộ (Local Server - Pop!_OS) | Máy chủ Đám mây (Cloud Server - RunPod) |
| :--- | :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 5060 Ti** | **NVIDIA GeForce RTX 3090 Ti** |
| **VRAM** | **16 GB GDDR7** | **24 GB GDDR6X** |
| **Kiến trúc GPU** | Ada Lovelace / Blackwell Refresh | Ampere |
| **Hệ điều hành** | Pop!_OS 22.04 LTS (Linux kernel 6.9.3) | Ubuntu 22.04 LTS (Containerized) |
| **CUDA Driver** | CUDA 12.8 / Driver 570.86.16 | CUDA 12.4 / Driver 550.54.15 |
| **Dịch vụ chạy ngầm** | `avir-kie-backend.service` (Port 8080)<br>`avir-kie-gpu.service` (Port 8000) | `qwen3_server` (Port 8000) |

---

## 2. Kiểm Toán Mức Tiêu Thụ Bộ Nhớ VRAM (VRAM Footprint)

Khi nạp mô hình và thực thi suy luận trên card **RTX 5060 Ti 16GB**:

| Trạng thái | VRAM Allocated (MB) | VRAM Reserved (MB) | % Dung lượng VRAM | Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **Khi nạp Base Model (4-bit NF4)** | 4,850 MB | 5,200 MB | 32.5% | Vô cùng tiết kiệm bộ nhớ |
| **Khi nạp LoRA v2 Adapter (PEFT)** | 5,120 MB | 5,600 MB | 35.0% | Adapter chỉ tăng thêm ~270MB |
| **Đỉnh điểm khi suy luận (Peak Inference)** | **6,225 MB (~6.2 GB)** | **6,796 MB (~6.8 GB)** | **42.5%** | **Còn dư hơn 9.2 GB VRAM trống** |
| **Kết luận khả thi** | Hệ thống hoàn toàn có thể chạy trên bất kỳ máy tính thương mại nào trang bị GPU từ **8GB VRAM trở lên** (RTX 3060, 4060, 3070, v.v.). |

---

## 3. Phân Rã Độ Trễ Suy Luận (Latency Breakdown)

### Mô hình 1: Qwen3-VL 8B LoRA v2 (End-to-End VLM)
* **Thời gian tiền xử lý ảnh (Resize aspect-ratio aware):** ~0.08 giây
* **Thời gian Vision Encoder (Patch embedding qua ViT):** ~1.45 giây
* **Thời gian sinh văn bản tự hồi quy (Autoregressive LLM Decoding ~180 tokens):** ~6.72 giây
* $\implies$ **Tổng độ trễ trung bình:** **8.25 giây / hóa đơn**.

### Mô hình 2: Two-Stage Hybrid (PaddleOCR + Spatial Grouping + Qwen2.5 7B)
* **Thời gian PaddleOCR (Text detection & recognition):** **0.79 giây** ⚡
* **Thời gian Spatial Line Grouping (Gom cụm tọa độ dải ngang trục Y):** **0.01 giây**
* **Thời gian Qwen2.5 7B trích xuất JSON (qua Ollama):** **6.69 giây**
* $\implies$ **Tổng độ trễ trung bình:** **7.49 giây / hóa đơn**.

---

## 4. Cơ Chế Chuyển Vùng Dự Phòng Tự Động (High-Availability Failover)
* Backend Gateway tự động kiểm tra `ping_ms` định kỳ tới máy chủ RunPod RTX 3090 Ti.
* Nếu máy chủ RunPod bị ngắt kết nối hoặc hết giờ thuê $\implies$ Hệ thống **tự động chuyển hướng không gián đoạn (Zero-Downtime Fallback)** sang máy chủ Pop!_OS RTX 5060 Ti cục bộ trong vòng **420ms**.
