@echo off
chcp 65001 >nul
title Stop AVIR-KIE API Gateway (Windows Host)
echo =====================================================================
echo   STOPPING AVIR-KIE GATEWAY & TAILSCALE FUNNEL
echo =====================================================================

echo [1/2] Disabling Tailscale Funnel on Port 8080...
tailscale funnel --https=443 off

echo [2/2] Killing any processes on Port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do (
    echo Killing PID %%a...
    taskkill /F /PID %%a
)

echo Done! Gateway stopped.
pause
