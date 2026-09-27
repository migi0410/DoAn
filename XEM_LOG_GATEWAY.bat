@echo off
chcp 65001 >nul
title Nhật Ký AVIR-KIE API Gateway (Real-Time Live Logs)
echo =====================================================================
echo   ĐANG THEO DÕI LOGS CỦA AVIR-KIE API GATEWAY (REAL-TIME)
echo   File: c:\Users\Admin\OneDrive\DoAn\logs\gateway.log
echo   Nhấn Ctrl + C để dừng theo dõi
echo =====================================================================
echo.
powershell -NoProfile -Command "Get-Content -Path 'c:\Users\Admin\OneDrive\DoAn\logs\gateway.log' -Wait -Tail 30"
pause
