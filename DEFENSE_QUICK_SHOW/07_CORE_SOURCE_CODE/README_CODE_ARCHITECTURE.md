# 💻 HƯỚNG DẪN KIẾN TRÚC MÃ NGUỒN (CORE SOURCE CODE ARCHITECTURE)

> **Mục đích:** Thư mục này tập hợp tất cả các file mã nguồn Python cốt lõi của đề tài **AVIR-KIE**, được phân nhóm khoa học theo từng chức năng để mở và trình bày trực tiếp cho Hội đồng phản biện.

---

## 🏗️ BẢN ĐỒ PHÂN NHÓM MÃ NGUỒN

```text
07_CORE_SOURCE_CODE/
│
├── 01_INFERENCE_ENGINE/          # Bộ suy luận mô hình thị giác - ngôn ngữ & Tiền xử lý ảnh
│   ├── inference_engine.py       # Bộ nạp Qwen3-VL 8B LoRA v2, inference pipeline (Transformers/vLLM)
│   ├── two_stage_spatial.py      # Module Two-Stage Hybrid: PaddleOCR + Phân cụm không gian Bounding Box
│   ├── document_processor.py     # Tiền xử lý OpenCV: Deskew (xoay thẳng góc), auto-crop viền, tăng tương phản
│   ├── yolo_cropper.py           # Bộ phát hiện 4 góc văn bản bằng YOLOv8 + Perspective Transform
│   └── serve_qwen3.py            # Script chạy service backend GPU phục vụ API inference
│
├── 02_TRAINING_PIPELINE/         # Pipeline tinh chỉnh mô hình đề xuất
│   ├── train_qwen3_vl_qlora_v2.py # Script SFTTrainer fine-tuning Qwen3-VL với 4-bit QLoRA (BitsAndBytes)
│   ├── prepare_dataset.py        # Chuẩn hóa nhãn JSONL sang định dạng hội thoại Vision đa dòng (Prompt v2)
│   ├── adapter_config.json       # Cấu hình siêu tham số LoRA (r=16, alpha=32, target modules)
│   └── chat_template.jinja       # Jinja2 template định dạng chat của Qwen3-VL
│
├── 03_EVALUATION_AND_BENCHMARKS/ # Bộ mã nguồn đo đạc thực nghiệm tự động
│   ├── benchmark_vietinvoice_1166.py # Đánh giá mô hình đề xuất trên 1,166 hóa đơn tập Test
│   ├── benchmark_mcocr_499.py    # Đánh giá mô hình trên 499 hóa đơn thực tế ngoài miền MCOCR
│   └── kie_full_evaluator.py     # Tính toán đầy đủ: Exact Match, Levenshtein NED, Macro-F1, Item Recall
│
├── 04_BASELINE_MODELS/           # Các mô hình đối sánh thực nghiệm
│   ├── baseline_layoutlmv3.py    # Huấn luyện và đánh giá LayoutLMv3 (Đa phương thức 2D)
│   ├── baseline_phobert_ner_v2.py# Huấn luyện và đánh giá PhoBERT (Mô hình ngôn ngữ thuần text tiếng Việt)
│   ├── baseline_craft_vietocr.py # Pipeline OCR 2 tầng truyền thống: CRAFT + VietOCR
│   ├── baseline_rule_based.py    # Baseline trích xuất bằng Biểu thức chính quy (Regex / Heuristics)
│   ├── LayoutLMv3_Kaggle_v2.ipynb# Jupyter Notebook huấn luyện LayoutLMv3 trên Kaggle GPU T4
│   └── PhoBERT_Kaggle_v2.ipynb   # Jupyter Notebook huấn luyện PhoBERT trên Kaggle
│
└── 05_API_AND_BACKEND/           # Tầng Web API phục vụ Demo thực tế
    ├── app.py                    # Khởi tạo FastAPI Server, CORS, Lifespan management
    ├── api.py                    # Các Router chính: /api/predict, /api/health, /api/gpu_status, history
    ├── ml_backend.py             # Bộ điều phối trung gian kết nối tiền xử lý và suy luận
    ├── database.py               # Quản lý lưu trữ lịch sử hóa đơn và kết quả KIE
    └── supabase_client.py        # Tích hợp Supabase Cloud Storage
```

---

## 🔍 ĐIỂM SÁNG KỸ THUẬT CẦN NHẤN MẠNH KHI THẦY/CÔ XEM CODE

### 1. File `01_INFERENCE_ENGINE/inference_engine.py` (Mô hình đề xuất cốt lõi)
* **Kỹ thuật:** Sử dụng `Qwen3VLForConditionalGeneration` kết hợp `PeftModel` để nạp adapter LoRA v2.
* **Cơ chế suy luận:** Tokenizer nạp `<|vision_start|><|image_pad|><|vision_end|>` kết hợp Prompt v2 ép đầu ra cấu trúc JSON nghiêm ngặt.
* **Giải thuật sửa sai số học:** Hàm `reconcile_item_math()` tự động đối chiếu \( \text{quantity} \times \text{unit\_price} == \text{total} \). Nếu phát hiện khuyết giá hoặc sai số do dòng quấn, thuật toán tự động nội suy từ tổng hóa đơn.

### 2. File `01_INFERENCE_ENGINE/two_stage_spatial.py` (Giải pháp đối sánh Two-Stage)
* **Kỹ thuật:** Kết hợp PaddleOCR tiếng Việt để nhận diện text và bounding box.
* **Thuật toán gom dòng:** Sử dụng khoảng cách trục Y ($\Delta y \le \theta_{line}$) và độ chồng lấn thẳng đứng (Vertical IoU) để ghép tên món dài rớt dòng vào đúng món mà không tạo dòng rác.

### 3. File `02_TRAINING_PIPELINE/train_qwen3_vl_qlora_v2.py` (Huấn luyện QLoRA 4-bit)
* **Cấu hình Quantization:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.
* **LoRA Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
* **Lợi ích:** Giảm yêu cầu VRAM từ **36GB xuống chỉ 6.2GB**, cho phép huấn luyện mô hình 8 tỷ tham số trên một card đồ họa phổ thông.

### 4. File `03_EVALUATION_AND_BENCHMARKS/kie_full_evaluator.py` (Đo lường học thuật)
* Đầy đủ 4 thước đo chuẩn quốc tế:
  1. **Macro-F1:** Đánh giá độ chính xác tổng hợp theo từng trường thông tin.
  2. **Line Item Recall:** Tỉ lệ phát hiện trúng các mặt hàng trong bảng kê chi tiết.
  3. **Exact Match (EM):** Tỉ lệ khớp 100% không sai một ký tự.
  4. **Normalized Edit Distance (NED):** Khoảng cách Levenshtein chuẩn hóa trên độ dài chuỗi ký tự.
