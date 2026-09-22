# AVIR-KIE Web Application Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting AVIR-KIE Demo Web Application" -ForegroundColor Green
Write-Host "  (Capstone AIP491 - FPT University HCMC)" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$rootDir\backend'; python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 2

Write-Host "[2/2] Starting Next.js 16 Frontend on http://localhost:3000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$rootDir\frontend'; npm run dev"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Services launched successfully:" -ForegroundColor Green
Write-Host "  - Backend API : http://localhost:8000 (Swagger: http://localhost:8000/docs)" -ForegroundColor White
Write-Host "  - Frontend UI : http://localhost:3000" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
