@echo off
chcp 65001 >nul
title AVIR-KIE API Gateway (Windows Host - Port 8080)
echo =====================================================================
echo   AVIR-KIE API GATEWAY (FastAPI on Port 8080)
echo   Local Address:      http://localhost:8080
echo   Public Tailscale:   https://migi.tail007aa8.ts.net
echo   Pop!_OS GPU Node:   http://100.80.138.26:8000
echo =====================================================================
cd /d "%~dp0"

echo [1/2] Enabling Tailscale Funnel on Port 8080...
tailscale funnel --bg --yes 8080

echo [2/2] Starting Uvicorn API Gateway on Port 8080...
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8080 --reload
pause
