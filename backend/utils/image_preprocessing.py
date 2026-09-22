import cv2
import numpy as np

def preprocess_for_ocr(image_path, max_dim=2000):
    """
    Tiền xử lý ảnh MCOCR để tối ưu cho OCR:
    1. Đọc ảnh.
    2. Resize nếu ảnh quá lớn để tránh nặng máy.
    3. Cân bằng sáng (CLAHE) để khử mờ/sấp bóng.
    4. Tăng độ nét (Sharpening).
    Trả về Numpy Array sẵn sàng truyền thẳng vào PaddleOCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Bỏ Resize để giữ nguyên tỷ lệ Bounding Box nguyên bản cho mô hình OCR và LayoutLM.
    # 2. Cân bằng sáng (CLAHE trên hệ màu LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE cho độ sáng (L-channel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Tăng độ nét (Unsharp Masking / Kernel)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_clahe, -1, kernel)
    
    # Có thể kết hợp nhẹ với Threshold hoặc Gray nếu cần, 
    # nhưng PaddleOCR v4 xử lý ảnh màu khá tốt sau khi nét & đủ sáng.
    return img_sharpened
