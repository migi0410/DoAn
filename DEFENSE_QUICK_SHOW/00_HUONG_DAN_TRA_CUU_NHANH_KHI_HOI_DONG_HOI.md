# 🎯 CẨM NANG TRA CỨU NHANH KHI BẢO VỆ ĐỒ ÁN TỐT NGHIỆP (AVIR-KIE)

> **Dành riêng cho nhóm tác giả:** Mở file này trên màn hình phụ hoặc cửa sổ riêng trong suốt buổi bảo vệ. Khi Thầy/Cô Hội đồng hỏi bất kỳ câu hỏi nào, lập tức mở file minh chứng tương ứng theo bảng dưới đây!

---

## 📌 BẢNG TRA CỨU "CÂU HỎI HỘI ĐỒNG $\rightarrow$ FILE CẦN MỞ"

| Câu hỏi của Thầy/Cô Hội Đồng | Mở ngay Thư mục / File này | Dẫn chứng quan trọng cần trả lời |
| :--- | :--- | :--- |
| **1. "Cho xem code huấn luyện QLoRA của mô hình đề xuất?"** | `07_CORE_SOURCE_CODE/02_TRAINING_PIPELINE/train_qwen3_vl_qlora_v2.py` | SFTTrainer của TRL kết hợp BitsAndBytes 4-bit NF4, PEFT LoRA $r=16, \alpha=32$, target 7 module chiếu tuyến tính. |
| **2. "Cho xem code suy luận (Inference), mô hình load thế nào?"** | `07_CORE_SOURCE_CODE/01_INFERENCE_ENGINE/inference_engine.py` | Nạp Qwen3-VL 8B + LoRA adapter, tiền xử lý ảnh và giải thuật tự động đối soát số học `reconcile_item_math()`. |
| **3. "Cho xem code Two-Stage Hybrid (PaddleOCR + Spatial BBox)?"** | `07_CORE_SOURCE_CODE/01_INFERENCE_ENGINE/two_stage_spatial.py` | PaddleOCR nhận diện bounding box tiếng Việt kết hợp thuật toán gom dòng không gian ($\Delta y \le \theta$ và vertical overlap). |
| **4. "Cho xem code tính các chỉ số F1, Recall, Levenshtein NED?"** | `07_CORE_SOURCE_CODE/03_EVALUATION_AND_BENCHMARKS/kie_full_evaluator.py` | Bộ hàm chuẩn hóa chuỗi, tính Macro-F1 theo từng trường, Exact Match và Normalized Edit Distance. |
| **5. "Cho xem file nhãn dữ liệu gốc (Ground Truth) định dạng ra sao?"** | `04_DATASET_STATISTICS_AND_EDA/vietinvoice_test_1166.jsonl` *(hoặc file nhẹ `vietinvoice_train_sample_100.jsonl`)* | Định dạng JSONL chuẩn hội thoại đa phương thức: `user` chứa `<image>` + prompt, `assistant` chứa JSON phân cấp. |
| **6. "Cho xem ảnh hóa đơn thực tế trong tập kiểm thử?"** | `04_DATASET_STATISTICS_AND_EDA/sample_images/` | Chứa 10 ảnh VietInvoice và 10 ảnh MCOCR thực tế (gấp nếp, mờ, bóng, nghiêng, giấy nhiệt phai mực). |
| **7. "Mô hình train thế nào? Cho xem log huấn luyện loss có giảm không?"** | `01_TRAINING_LOGS/train_v2.log` | Mở log, kéo xuống cuối: Loss giảm từ **1.85** xuống **~0.082**. Train 3 epochs, QLoRA 4-bit trên RTX 5060 Ti / 3090 Ti. |
| **8. "Số liệu 93.35% F1 hay 98.63% Recall này ở đâu ra? File kết quả chi tiết đâu?"** | `02_BENCHMARK_RESULTS_CSV/vietinvoice_1166_benchmark_results.csv` | File CSV chứa kết quả chi tiết từng dòng của **1,166 hóa đơn test**. Có đủ cột Ground Truth, Pred, F1, Levenshtein, Exact Match. |
| **9. "Thử nghiệm trên dữ liệu thực tế ngoài đời (MCOCR) thế nào?"** | `02_BENCHMARK_RESULTS_CSV/mcocr_499_benchmark_results.csv` | File CSV đánh giá trên **499 ảnh hóa đơn thực tế MCOCR** (out-of-domain). Đạt **74.1% F1** và **88.3% Item Recall**. |
| **10. "Số liệu giữa báo cáo Word, Slide và Code có đồng nhất không?"** | `02_BENCHMARK_RESULTS_CSV/OFFICIAL_VERIFIED_NUMBERS_AUDIT.md` | Bảng kiểm kê chính thức: Khớp 100% từng con số giữa Report LaTeX/Word, Slide và kết quả chạy thực nghiệm. |
| **11. "So sánh mô hình đề xuất với PhoBERT, LayoutLMv3, DeepSeek, Regex?"** | `03_MODEL_COMPARISON_EVIDENCE/BIEU_DO_SO_SANH_CAC_MO_HINH.md` | Bảng đối đầu 6 mô hình: Qwen3 LoRA v2 (93.35%) áp đảo PhoBERT (74.2%), LayoutLMv3 (84.1%), DeepSeek (66.9%), Regex (52.4%). |
| **12. "Hệ thống chạy trên GPU gì? Tốn bao nhiêu VRAM? Độ trễ (latency) bao lâu?"** | `06_SYSTEM_AND_HARDWARE_SPECS/GPU_AND_LATENCY_AUDIT.md` | GPU: RTX 5060 Ti 16GB GDDR7 (Pop!_OS) & RTX 3090 Ti (RunPod). VRAM chỉ tốn **6.2 GB**. Độ trễ: **8.25s** (VLM), **7.5s** (Two-Stage). |

---

## 💎 BẢNG SỐ LIỆU VÀNG CẦN NHỚ NẰM LÒNG (CHEAT SHEET)

* **Tên đề tài:** Nghiên cứu Trích Xuất Thông Tin Hóa Đơn Bán Lẻ Tiếng Việt sử dụng Mô Hình Thị Giác - Ngôn Ngữ Tinh Chỉnh (AVIR-KIE).
* **Mô hình cốt lõi:** **Qwen3-VL (8B)** tinh chỉnh bằng **4-bit QLoRA** với **Prompt v2 (Cấu trúc phân cấp đa dòng)**.
* **Tập dữ liệu:**
  - Tổng số mẫu: **13,690 hóa đơn**.
  - Chia tập: Train = **10,000** | Val = **2,025** | Test VietInvoice = **1,166** | Test MCOCR thực tế = **499**.
* **Hiệu năng trên tập Test VietInvoice (1,166 mẫu):**
  - **Macro-F1:** **93.35%** (Cao nhất trong mọi kiến trúc).
  - **Line Item Recall:** **98.63%** (Khả năng bắt trọn từng món hàng).
  - **Exact Match:** **86.87%**.
  - **Độ trễ trung bình:** **8.25 giây / hóa đơn**.
* **Hiệu năng trên tập MCOCR thực tế (499 mẫu ngoài miền):**
  - **Macro-F1:** **74.10%** (Vượt xa LayoutLMv3 chỉ đạt 61.2%).
  - **Item Recall:** **88.30%**.
* **Mức tiêu thụ tài nguyên phần cứng:**
  - VRAM chiếm dụng khi chạy thực tế: **6,225 MB (~6.2 GB)** $\implies$ Chạy mượt mà trên card đồ họa phổ thông 8GB/16GB.
