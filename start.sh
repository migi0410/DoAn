#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define directories
export LABEL_STUDIO_BASE_DATA_DIR=/app/data
mkdir -p /app/data

# Resolve Django CSRF proxy issues inside Hugging Face Spaces (Stateless iframe/reverse proxy)
export USE_X_FORWARDED_HOST=true
export USE_X_FORWARDED_PORT=true
export SECURE_PROXY_SSL_HEADER="HTTP_X_FORWARDED_PROTO,https"
export CSRF_TRUSTED_ORIGINS="https://*.hf.space,https://*.huggingface.co"

# Configure Cookies to work inside Iframe (Cross-site Cookie SameSite=None; Secure)
export CSRF_COOKIE_SAMESITE="None"
export CSRF_COOKIE_SECURE="1"
export SESSION_COOKIE_SAMESITE="None"
export SESSION_COOKIE_SECURE="1"

# Increase direct file upload limits for large receipt batches
export DATA_UPLOAD_MAX_NUMBER_FILES=10000

# Start the PaddleOCR ML Backend in the background on port 9090
echo "🚀 Starting PaddleOCR ML Backend on port 9090..."
python ml_backend.py 2>&1 &

# Wait a couple of seconds to make sure FastAPI is up
sleep 3

# Pre-configure Admin Account for Label Studio
export LABEL_STUDIO_USERNAME="admin@avir.kie"
export LABEL_STUDIO_PASSWORD="AdminPassword2026!"

# Start Label Studio on port 7860 (Hugging Face standard port)
echo "🧾 Starting Label Studio on port 7860..."
label-studio start --port 7860 --host 0.0.0.0
