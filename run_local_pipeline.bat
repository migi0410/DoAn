@echo off
chcp 65001 >nul
echo ==================================================
echo   KHOI DONG HE THONG LOCAL HYBRID PIPELINE (OLLAMA)
echo ==================================================
echo.
echo Dang kiem tra model moondream tren may...
ollama list | findstr "moondream" >nul
if %errorlevel% neq 0 (
    echo Model chua co san hoac dang tai do dang.
    echo Vui long cho xiu de he thong tai ve nhe - Khoang 1.7GB...
    ollama pull moondream
) else (
    echo Model moondream da san sang!
)

echo.
echo ==================================================
echo Bat dau chay Pipeline xu ly 200 tam hoa don...
echo ==================================================
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set PYTHONIOENCODING=utf-8
python -u utils\hybrid_label_pipeline.py

echo.
echo ==================================================
echo HOAN THANH TOAN BO QUA TRINH!
echo ==================================================
pause
