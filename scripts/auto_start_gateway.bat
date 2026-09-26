@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

:: Check if port 8080 is already active first
netstat -ano | findstr :8080 | findstr LISTENING >nul
if %ERRORLEVEL% equ 0 (
    exit /b 0
)

:: Wait 5 seconds for network and Tailscale to initialize on system boot
ping -n 6 127.0.0.1 >nul

:: Double check if port 8080 became active while waiting
netstat -ano | findstr :8080 | findstr LISTENING >nul
if %ERRORLEVEL% equ 0 (
    exit /b 0
)

:: Ensure logs directory exists
if not exist "logs" mkdir logs

:: 1. Enable Tailscale Funnel on Port 8080
if exist "C:\Program Files\Tailscale\tailscale.exe" (
    "C:\Program Files\Tailscale\tailscale.exe" funnel --bg --yes 8080 >> "logs\gateway.log" 2>&1
)

:: 2. Start Uvicorn API Gateway and log output
echo [%date% %time%] Starting AVIR-KIE API Gateway on Port 8080... >> "logs\gateway.log"
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8080 >> "logs\gateway.log" 2>&1
