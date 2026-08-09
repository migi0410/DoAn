import cv2
import numpy as np
from typing import List, Dict, Any

class ImagePreprocessor:

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return image
            
        angles = []
        for line in lines:
            if line.ndim == 2:
                x1, y1, x2, y2 = line[0]
            elif line.ndim == 1 and len(line) == 4:
                x1, y1, x2, y2 = line
            else:
                x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle > 90: angle -= 180
            elif angle < -90: angle += 180
            angles.append(angle)
            
        counts, bins = np.histogram(angles, bins=180, range=(-90, 90))
        max_bin = np.argmax(counts)
        dominant_angle = (bins[max_bin] + bins[max_bin + 1]) / 2
        
        # Hough lines cannot determine text orientation (0 vs 180, 90 vs -90).
        # We only want to fix slight skews, not flip the image randomly by 90 degrees.
        if dominant_angle > 45:
            dominant_angle -= 90
        elif dominant_angle < -45:
            dominant_angle += 90
            
        if abs(dominant_angle) < 0.5:
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def enhance(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w > 1500:
            ratio = 1500.0 / w
            image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
            
        # 1. Gentle Unsharp Masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # 2. Mild Contrast Enhancement via LAB color space (preserves colors)
        try:
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception:
            # Fallback if image is somehow grayscale
            return sharpened

    @staticmethod
    def crop_document(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        orig = image.copy()
        ratio = image.shape[0] / 500.0
        res = cv2.resize(image, (int(image.shape[1] / ratio), 500))
        gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_res, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = res.shape[0] * res.shape[1]
        valid = [c for c in contours if img_area * 0.1 < cv2.contourArea(c) < img_area * 0.95]
        if not valid:
            edged = cv2.Canny(blurred, 50, 150)
            edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if img_area * 0.1 < cv2.contourArea(c) < img_area * 0.95]
            if not valid:
                return image
        largest_cnt = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        x, y, w, h = (int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio))
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2 * pad)
        h = min(image.shape[0] - y, h + 2 * pad)
        return image[y:y + h, x:x + w]

    @staticmethod
    def process_all(image: np.ndarray) -> np.ndarray:
        cropped = ImagePreprocessor.crop_document(image)
        deskewed = ImagePreprocessor.deskew(cropped)
        enhanced = ImagePreprocessor.enhance(deskewed)
        return enhanced

class TextPreprocessor:

    @staticmethod
    def sort_reading_order(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not boxes:
            return []
        enriched_boxes = []
        for item in boxes:
            box = item['box']
            if len(box) == 4 and isinstance(box[0], (int, float)):
                # Format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
            else:
                # Format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            h = y2 - y1
            enriched_boxes.append({'original': item, 'cx': cx, 'cy': cy, 'h': h, 'x1': x1, 'y1': y1})
        enriched_boxes.sort(key=lambda x: x['cy'])
        if len(enriched_boxes) > 0:
            median_h = np.median([b['h'] for b in enriched_boxes])
        else:
            median_h = 10
        y_threshold = median_h * 0.5
        lines = []
        current_line = []
        for item in enriched_boxes:
            if not current_line:
                current_line.append(item)
            else:
                avg_cy = sum((b['cy'] for b in current_line)) / len(current_line)
                if abs(item['cy'] - avg_cy) <= y_threshold:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
        if current_line:
            lines.append(current_line)
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda x: x['x1'])
            for item in line:
                sorted_boxes.append(item['original'])
        return sorted_boxes
if __name__ == '__main__':
    print('Preprocessing module loaded.')
