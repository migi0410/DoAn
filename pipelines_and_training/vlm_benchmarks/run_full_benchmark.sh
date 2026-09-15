#!/bin/bash
set -e

cd /home/haderax/DoAn/official_benchmark
PYTHON="/home/haderax/qwen_train_env/bin/python"
LOG="/home/haderax/DoAn/official_benchmark/benchmark_run.log"

echo "==========================================================================" >> "$LOG"
echo "🚀 BẮT ĐẦU CHUỖI BENCHMARK TOÀN DIỆN HỌ QWEN3-VL (1,166 MẪU)" >> "$LOG"
echo "Thời gian bắt đầu: $(date)" >> "$LOG"
echo "==========================================================================" >> "$LOG"

# 1. Chạy Qwen3-VL LoRA v2 (Prompt v2)
echo ">>> [1/3] Bắt đầu đánh giá Qwen3-VL LoRA v2 (Prompt v2)..." >> "$LOG"
"$PYTHON" /home/haderax/DoAn/official_benchmark/run_benchmark_qwen_matrix.py --models lora_v2 >> "$LOG" 2>&1
echo ">>> [1/3] Hoàn thành Qwen3-VL LoRA v2 lúc: $(date)" >> "$LOG"

# 2. Chạy Qwen3-VL Base v2 (Prompt v2) qua Ollama
echo ">>> [2/3] Bắt đầu đánh giá Qwen3-VL Base v2 (Prompt v2) qua Ollama..." >> "$LOG"
"$PYTHON" /home/haderax/DoAn/official_benchmark/run_benchmark_qwen_matrix.py --models base_v2 >> "$LOG" 2>&1
echo ">>> [2/3] Hoàn thành Qwen3-VL Base v2 lúc: $(date)" >> "$LOG"

# 3. Dừng serve_qwen3.py để giải phóng 100% VRAM cho LoRA v1
echo ">>> [3/3] Giải phóng VRAM để đánh giá Qwen3-VL LoRA v1 (Prompt v1)..." >> "$LOG"
pkill -f serve_qwen3.py || true
sleep 3
"$PYTHON" /home/haderax/DoAn/official_benchmark/run_benchmark_qwen_matrix.py --models lora_v1 >> "$LOG" 2>&1
echo ">>> [3/3] Hoàn thành Qwen3-VL LoRA v1 lúc: $(date)" >> "$LOG"

echo "==========================================================================" >> "$LOG"
echo "🎉 TẤT CẢ 7 MÔ HÌNH ĐÃ HOÀN TẤT BENCHMARK 100%!" >> "$LOG"
echo "Thời gian kết thúc: $(date)" >> "$LOG"
echo "==========================================================================" >> "$LOG"
