#!/bin/bash
# Setup isolated venv for MiniCPM-V (needs transformers==4.40, conflicts with Qwen2-VL>=4.46)
set -e
VENV=/workspace/minicpm_env
echo "Creating venv at $VENV..."
python3 -m venv $VENV
source $VENV/bin/activate
pip install -q --upgrade pip
pip install -q \
    "transformers==4.40.0" \
    "peft>=0.10.0" \
    "Pillow" \
    "fastapi" \
    "uvicorn" \
    "accelerate" \
    "sentencepiece" \
    "huggingface_hub" \
    "numpy>=1.24,<2.0"
# Install torch (use pre-installed if available)
python3 -c "import torch; print('torch already ok:', torch.__version__)" 2>/dev/null || \
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
echo ""
echo "MiniCPM venv ready at $VENV"
echo "Start server with:"
echo "  MINICPM_MODEL_DIR=/workspace/minicpm_v_lora_official $VENV/bin/python3 /workspace/DoAn/src/backend/minicpm_server.py"
