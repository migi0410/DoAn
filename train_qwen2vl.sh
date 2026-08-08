#!/bin/bash
# train_qwen2vl.sh
# Train Qwen2-VL-2B-Instruct LoRA với CLEAN dataset
# 1 epoch, train xong la dung, khong chay them gi ca
#
# Cach dung: bash train_qwen2vl.sh

set -e

DATASET_PATH="/workspace/DoAn/data/CLEAN_TRAIN_DATASET/train.jsonl"
OUTPUT_DIR="/workspace/qwen2_vl_lora_v2"
MODEL="Qwen/Qwen2-VL-2B-Instruct"

echo "============================================"
echo "  Qwen2-VL LoRA Training"
echo "  Dataset: $DATASET_PATH"
echo "  Output:  $OUTPUT_DIR"
echo "  Epochs:  1"
echo "============================================"

# Kiem tra dataset ton tai
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Make sure you ran: git pull origin main"
    exit 1
fi

echo "Dataset found: $(wc -l < $DATASET_PATH) samples"
echo "Starting training..."
echo ""

swift sft \
    --model "$MODEL" \
    --model_type qwen2_vl \
    --template_type qwen2_vl \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_type lora \
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
    --report_to tensorboard \
    --logging_steps 10

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint saved to: $OUTPUT_DIR"
echo "  DO NOT run anything else - check logs first"
echo "============================================"
