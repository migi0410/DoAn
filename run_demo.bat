@echo off
title AVIR-KIE Web Application Launcher
echo ========================================================
echo   Starting AVIR-KIE Demo Web Application
echo   (Capstone AIP491 - FPT University HCMC)
echo ========================================================
echo.

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "AVIR Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 >nul

echo [2/2] Starting Next.js 16 Frontend on http://localhost:3000 ...
start "AVIR Frontend (Next.js)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================================
echo   Services are starting up:
echo   - Backend API : http://localhost:8000 (Docs: http://localhost:8000/docs)
echo   - Frontend UI : http://localhost:3000
echo ========================================================
echo.
pause
