#!/bin/bash
# train_all_vlm.sh
# Train Qwen2-VL + MiniCPM-V-2_6 tuan tu voi CLEAN dataset
# 1 epoch moi model, train xong la dung, khong chay them gi ca
#
# Cach dung: bash train_all_vlm.sh
# Log duoc ghi tu dong vao: /workspace/logs/

set -e
set -o pipefail  # Ensure errors in pipes (e.g. tee) are caught

DATASET_PATH="/workspace/DoAn/data/OFFICIAL_DATASET/train.jsonl"
VAL_DATASET_PATH="/workspace/DoAn/data/OFFICIAL_DATASET/val.jsonl"
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

# ============================================
# KIEM TRA DATASET
# ============================================
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run: git pull origin main"
    exit 1
fi

# Tao symlink images tai CWD (/workspace/DoAn/images)
# Vi Swift resolve path tuong doi tu thu muc chay script, khong phai tu JSONL
IMAGES_SRC="/workspace/FINAL_RUNPOD_DATASET/images"
IMAGES_LINK="/workspace/DoAn/images"
if [ ! -L "$IMAGES_LINK" ] && [ ! -d "$IMAGES_LINK" ]; then
    if [ -d "$IMAGES_SRC" ]; then
        ln -sf "$IMAGES_SRC" "$IMAGES_LINK"
        echo "Symlink created: $IMAGES_LINK -> $IMAGES_SRC"
    else
        echo "ERROR: Images not found at $IMAGES_SRC"
        exit 1
    fi
else
    echo "Images dir OK: $IMAGES_LINK"
fi

SAMPLE_COUNT=$(wc -l < "$DATASET_PATH")
echo "============================================"
echo "  Dataset: $DATASET_PATH"
echo "  Samples: $SAMPLE_COUNT"
echo "============================================"
echo ""

# ============================================
# KIEM TRA WANDB
# ============================================
if ! python -c "import wandb" 2>/dev/null; then
    echo "[WandB] Not installed. Installing..."
    pip install wandb -q
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "[WandB] WANDB_API_KEY not set. Logging to tensorboard only."
    REPORT_TO="tensorboard"
else
    echo "[WandB] API key found. Logging to wandb + tensorboard."
    export WANDB_PROJECT="avir-kie-vlm"
    REPORT_TO="tensorboard wandb"
fi
echo ""

# ============================================
# BUOC 1: TRAIN QWEN2-VL-2B-INSTRUCT
# ============================================
echo "[1/2] Training Qwen2-VL-2B-Instruct..."
echo "      Output: /workspace/qwen2_vl_lora_v2"
echo ""

swift sft \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset "$DATASET_PATH" \
    --val_dataset "$VAL_DATASET_PATH" \
    --output_dir /workspace/qwen2_vl_lora_official \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --dataloader_num_workers 2 \
    --report_to $REPORT_TO \
    --logging_steps 10 \
    2>&1 | tee "$LOG_DIR/qwen2vl_train.log"

echo ""
echo "[1/2] Qwen2-VL training DONE."
echo "      Log saved to: $LOG_DIR/qwen2vl_train.log"
echo ""

# ============================================
# BUOC 2: TRAIN MINICPM-V-2_6
# ============================================
echo "[2/2] Training MiniCPM-V-2_6..."
echo "      Output: /workspace/minicpm_v_lora_v2"
echo ""

swift sft \
    --model openbmb/MiniCPM-V-2_6 \
    --dataset "$DATASET_PATH" \
    --val_dataset "$VAL_DATASET_PATH" \
    --output_dir /workspace/minicpm_v_lora_official \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --dataloader_num_workers 2 \
    --report_to $REPORT_TO \
    --logging_steps 10 \
    2>&1 | tee "$LOG_DIR/minicpm_train.log"

echo ""
echo "[2/2] MiniCPM-V training DONE."
echo "      Log saved to: $LOG_DIR/minicpm_train.log"
echo ""

echo "============================================"
echo "  ALL TRAINING COMPLETE"
echo "  Qwen2-VL:  /workspace/qwen2_vl_lora_official"
echo "  MiniCPM-V: /workspace/minicpm_v_lora_official"
echo "  Logs:      $LOG_DIR/"
echo "============================================"
