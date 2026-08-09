#!/bin/bash
# setup_new_pod.sh
# Chay lan dau tren pod moi sau khi migration
# Usage: bash setup_new_pod.sh

set -e

echo "======================================"
echo "  SETUP NEW RUNPOD INSTANCE"
echo "======================================"

# 1. Clone repo
if [ ! -d "/workspace/DoAn" ]; then
    echo "[1/5] Cloning repo..."
    git clone https://github.com/migi0410/DoAn.git /workspace/DoAn
else
    echo "[1/5] Repo exists, pulling latest..."
    cd /workspace/DoAn && git pull origin main
fi

cd /workspace/DoAn

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install ms-swift wandb python-Levenshtein -q

# 3. WandB login
echo "[3/5] WandB setup..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  WARNING: WANDB_API_KEY not set."
    echo "  Run: export WANDB_API_KEY=your_key"
else
    wandb login "$WANDB_API_KEY"
    echo "  WandB logged in."
fi

# 4. Check images
echo "[4/5] Checking images..."
TRAIN_IMAGES="/workspace/FINAL_RUNPOD_DATASET/images"
MCOCR_IMAGES="/workspace/bench_images"

if [ -d "$TRAIN_IMAGES" ]; then
    COUNT=$(ls "$TRAIN_IMAGES" | wc -l)
    echo "  Training images: $COUNT files at $TRAIN_IMAGES"
else
    echo "  WARNING: Training images NOT FOUND at $TRAIN_IMAGES"
    echo "  Please upload: scp -r local_images/ root@<pod>:/workspace/FINAL_RUNPOD_DATASET/images/"
fi

if [ -d "$MCOCR_IMAGES" ]; then
    COUNT=$(ls "$MCOCR_IMAGES" | wc -l)
    echo "  MCOCR images: $COUNT files at $MCOCR_IMAGES"
else
    echo "  WARNING: MCOCR images NOT FOUND at $MCOCR_IMAGES"
fi

# 5. Create symlink for training
echo "[5/5] Creating image symlink..."
IMAGES_LINK="/workspace/DoAn/images"
if [ -d "$TRAIN_IMAGES" ] && [ ! -L "$IMAGES_LINK" ] && [ ! -d "$IMAGES_LINK" ]; then
    ln -sf "$TRAIN_IMAGES" "$IMAGES_LINK"
    echo "  Symlink created: $IMAGES_LINK -> $TRAIN_IMAGES"
else
    echo "  Symlink OK or images not ready yet."
fi

echo ""
echo "======================================"
echo "  SETUP DONE"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Upload images if missing (see WARNING above)"
echo "  2. export WANDB_API_KEY=your_key"
echo "  3. bash train_all_vlm.sh"
echo "======================================"
