FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/app

WORKDIR /app

# Install system dependencies for OpenCV and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and startup script
COPY ml_backend.py start.sh /app/

# Make startup script executable
RUN chmod +x /app/start.sh

# Create data directory and set wide permissions (required by Hugging Face sandboxed non-root user)
RUN mkdir -p /app/data && chmod -R 777 /app

EXPOSE 7860

# Run the startup script
CMD ["/app/start.sh"]
