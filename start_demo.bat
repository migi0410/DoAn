@echo off
chcp 65001 >nul
title AVIR-KIE Demo Launcher - Document Intelligence

echo =====================================================================
echo        AVIR-KIE: VISION-LANGUAGE MODEL DOCUMENT INTELLIGENCE
echo                  KHOA HOC VA CONG NGHE - DEMO KHOA LUAN
echo =====================================================================
echo.

cd /d "%~dp0"

echo [1/3] Khoi dong Backend API (FastAPI)...
start "AVIR-KIE Backend" /min cmd /c "cd /d %~dp0backend && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

echo [2/3] Khoi dong Frontend Web UI (Next.js - Port 3000)...
start "AVIR-KIE Frontend" /min cmd /c "cd /d %~dp0frontend && npm run start -- -p 3000"

echo [3/3] Dang khoi tao dich vu...
timeout /t 3 /nobreak >nul

echo.
echo =====================================================================
echo  DA KHOI DONG HE THONG DEMO THANH CONG!
echo  - Desktop Web: http://localhost:3000
echo  - Mobile Web:  http://192.168.100.29:3000
echo =====================================================================
echo.
echo Dang mo trinh duyet...
start http://localhost:3000

pause
