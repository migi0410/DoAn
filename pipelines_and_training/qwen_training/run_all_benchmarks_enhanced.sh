#!/bin/bash
set -e

PYTHON="/home/haderax/qwen_train_env/bin/python"
LOG="/home/haderax/DoAn/benchmark_all_enhanced.log"

echo "==========================================================================" >> "$LOG"
echo "🚀 BẮT ĐẦU BENCHMARK TOÀN DIỆN CHO QWEN3-VL LORA V2 (ENHANCED)" >> "$LOG"
echo "Thời gian khởi động: $(date)" >> "$LOG"
echo "==========================================================================" >> "$LOG"

# 1. Đánh giá trên Official Benchmark (1,166 mẫu)
echo ">>> [1/2] Đang chạy Official Benchmark (1,166 mẫu)..." >> "$LOG"
cd /home/haderax/DoAn/official_benchmark
"$PYTHON" run_benchmark_qwen_matrix.py --models lora_v2_enhanced >> "$LOG" 2>&1
echo ">>> [1/2] Hoàn thành Official Benchmark lúc: $(date)" >> "$LOG"

# 2. Đánh giá trên MCOCR Benchmark (499 mẫu)
echo ">>> [2/2] Đang chạy MCOCR Benchmark (499 mẫu)..." >> "$LOG"
cd /home/haderax/DoAn/mcocr_benchmark
"$PYTHON" run_benchmark_mcocr_vlms.py --models lora_v2_enhanced >> "$LOG" 2>&1
echo ">>> [2/2] Hoàn thành MCOCR Benchmark lúc: $(date)" >> "$LOG"

echo "==========================================================================" >> "$LOG"
echo "🎉 HOÀN TẤT TOÀN BỘ 2 BỘ DỮ LIỆU (1,166 + 499 = 1,665 MẪU)!" >> "$LOG"
echo "Thời gian kết thúc: $(date)" >> "$LOG"
echo "==========================================================================" >> "$LOG"
