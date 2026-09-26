# ⚙️ MINH CHỨNG THÔNG SỐ HUẤN LUYỆN (TRAINING HYPERPARAMETERS & LOGS)

## 1. Thông Tin Kiến Trúc Huấn Luyện
* **Mô hình nền tảng (Base Model):** `Qwen/Qwen3-VL-8B-Instruct`
* **Kỹ thuật tối ưu:** **Parameter-Efficient Fine-Tuning (PEFT)** với **QLoRA 4-bit (NormalFloat4 - NF4)**.
* **Hệ điều hành & Môi trường:** Pop!_OS 22.04 LTS Linux, Python 3.12, PyTorch 2.11, CUDA 12.8, BitsAndBytes 0.43.3.
* **Phần cứng:** 
  - Máy chủ cục bộ: **NVIDIA GeForce RTX 5060 Ti 16GB GDDR7**
  - Máy chủ điện toán đám mây: **NVIDIA GeForce RTX 3090 Ti 24GB**

## 2. Bảng Siêu Tham Số Huấn Luyện Chi Tiết

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Rationale / Giải trình kỹ thuật |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | `16` | Cân bằng hoàn hảo giữa khả năng học biểu diễn và kích thước adapter |
| **LoRA Alpha ($\alpha$)** | `32` | Tỉ lệ scaling $\alpha / r = 2.0$, giúp cập nhật trọng số ổn định |
| **LoRA Dropout** | `0.05` | Chống overfitting trên các mẫu hóa đơn in nhiệt đặc thù |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Can thiệp toàn diện cả Attention Projection và MLP Feed-Forward |
| **Quantization Type** | `4-bit (NF4, Double Quantization)` | Nén trọng số cơ sở 8B từ ~16GB xuống chỉ còn ~4.8GB trong VRAM |
| **Compute Dtype** | `torch.bfloat16` | Giữ độ chính xác số học cao, chống tràn số |
| **Learning Rate** | `1e-4` ($0.0001$) | Tốc độ hội tụ tối ưu cho QLoRA trên Vision-Language Models |
| **LR Scheduler** | `CosineAnnealingLR` | Hạ dần learning rate về $1e-6$ ở cuối quá trình huấn luyện |
| **Warmup Ratio** | `0.05` (5% tổng steps) | Ổn định gradients ở những bước đầu tiên |
| **Batch Size per GPU** | `4` | Phù hợp dung lượng VRAM 16GB |
| **Gradient Accumulation** | `4` | Effective Batch Size = $4 \times 4 = \mathbf{16}$ |
| **Số Epochs** | `3.0` | Đạt điểm cực tiểu loss mà không bị quá khớp |
| **Optimizer** | `PagedAdamW 8-bit` | Chống lỗi Out-of-Memory khi VRAM đạt đỉnh |
| **Max Sequence Length** | `2,048 tokens` | Chứa trọn vẹn toàn bộ hóa đơn dài tới 40 món hàng |

## 3. Quá Trình Hội Tụ Của Loss (Đoạn trích từ `train_v2.log`)
* **Step 0:** `Training Loss = 1.8462`
* **Step 250:** `Training Loss = 0.5418`
* **Step 500:** `Training Loss = 0.2842`
* **Step 1000:** `Training Loss = 0.1415`
* **Step 1875 (Kết thúc):** `Training Loss = 0.0824` $\implies$ Mô hình hội tụ mượt mà, không xảy ra hiện tượng Gradient Explosion hay Divergence.
