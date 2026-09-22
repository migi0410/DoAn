import cv2
import numpy as np

def order_points(pts):
    # Initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # set up the output dimensions
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def enhance_illumination(image):
    # Chuyển ảnh sang không gian màu LAB để phân tách kênh độ sáng (Lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) lên kênh độ sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Gộp các kênh lại
    limg = cv2.merge((cl, a, b))
    
    # Chuyển lại sang BGR
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def process_document(image_path, output_path=None):
    """
    Nhận vào 1 ảnh, cắt nền (nếu tìm thấy), cân bằng sáng và lưu/trả về.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc được ảnh từ {image_path}")
        
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    
    # Resize để xử lý nhanh hơn và tìm viền dễ hơn
    import imutils
    image = imutils.resize(image, height=500)

    # Chuyển grayscale, blur, và tìm edge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Tìm các Contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # Lặp qua các contour để tìm tứ giác
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # Nếu tìm thấy khung hình chữ nhật (Tức là mặt hóa đơn rõ ràng)
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        # Nếu không tìm thấy viền vuông (Do ảnh chụp sát rìa, bị che...), 
        # thì bỏ qua bước cắt, chỉ giữ ảnh gốc
        warped = orig.copy()

    # Cân bằng sáng (CLAHE) để khử bóng tay người, vùng sáng tối không đều
    enhanced = enhance_illumination(warped)
    
    if output_path:
        cv2.imwrite(output_path, enhanced)
        return output_path
        
    return enhanced
