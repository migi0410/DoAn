@echo off
:: Thiết lập mã hóa UTF-8 cho Command Prompt để hiển thị tiếng Việt không lỗi font
chcp 65001 > nul

echo ===================================================================
echo        DỰ ÁN AVIR-KIE - KHỞI CHẠY LABEL STUDIO TỰ ĐỘNG CỤC BỘ
echo ===================================================================
echo.

:: 1. Kiểm tra xem Python đã được cài đặt và cấu hình PATH chưa
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Không tìm thấy Python trên máy tính của bạn!
    echo Vui lòng tải và cài đặt Python (Khuyên dùng bản 3.8 đến 3.11).
    echo CHÚ Ý: Hãy tích chọn mục "Add Python to PATH" khi cài đặt.
    echo.
    pause
    exit /b
)

:: 2. Thiết lập biến môi trường cho phép đọc file ảnh local (Cực kỳ quan trọng)
:: Cú pháp viết liền && tránh khoảng trắng dư thừa làm lỗi biến
(set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true)

:: 3. Kiểm tra xem thư viện label-studio đã được cài đặt trong Python chưa
python -c "import label_studio" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Thư viện 'label-studio' chưa có trên máy của bạn.
    echo Đang tự động tải xuống và cài đặt (Quá trình này mất khoảng 1-2 phút)...
    echo.
    python -m pip install label-studio
    
    :: Kiểm tra xem cài đặt thành công không
    python -c "import label_studio" >nul 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Cài đặt Label Studio thất bại! Vui lòng kiểm tra lại kết nối mạng.
        pause
        exit /b
    )
    echo.
    echo [SUCCESS] Cài đặt Label Studio thành công!
    echo.
)

:: 4. Khởi chạy Label Studio sử dụng Module Python di động (tránh lỗi PATH)
echo [INFO] Đang khởi chạy máy chủ Label Studio...
echo.
echo ===================================================================
echo  BƯỚC TIẾP THEO:
echo  1. Mở trình duyệt web của bạn và truy cập: http://localhost:8080
echo  2. Thực hiện gán nhãn theo hướng dẫn trong file README.md
echo  3. TẮT MÁY CHỦ: Nhấn giữ tổ hợp phím [Ctrl + C] tại cửa sổ này.
echo ===================================================================
echo.

python -c "from label_studio.server import main; main()"

pause
