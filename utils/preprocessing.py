import cv2
import numpy as np
from typing import List, Dict, Any

class ImagePreprocessor:
    """
    Handles image-level preprocessing for receipts before sending to OCR.
    Pipeline: Deskew -> Blur -> CLAHE -> Adaptive Threshold -> Unsharp Mask.
    """
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Detects the skew angle of the receipt text using Hough Line Transform and rotates it back to straight.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near horizontal or near vertical lines
                if -45 < angle < 45:
                    angles.append(angle)
                elif angle > 45:
                    angles.append(angle - 90)
                elif angle < -45:
                    angles.append(angle + 90)
        
        if not angles:
            return image
            
        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        # Use replicate border to avoid glaring white borders
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def enhance(image: np.ndarray) -> np.ndarray:
        """
        Applies standard OCR enhancements: Blur -> CLAHE -> Threshold -> Sharpen
        """
        # 1. Resize for speed if image is massive (max width 1500px)
        h, w = image.shape[:2]
        if w > 1500:
            ratio = 1500.0 / w
            image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 2. Noise reduction (Gaussian Blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 4. Binarization (Adaptive Thresholding)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        
        # 5. Sharpening (Unsharp Masking)
        blur_for_unsharp = cv2.GaussianBlur(binary, (5, 5), 1.0)
        sharpened = float(2.0) * binary - float(1.0) * blur_for_unsharp
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        
        # Return as 3 channel BGR for downstream models (LayoutLM / OCR expect 3 channels)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
    @staticmethod
    def crop_document(image: np.ndarray) -> np.ndarray:
        """
        Finds the largest document-like contour and crops the image to its bounding box.
        Uses adaptive thresholding + morphological closing for robust detection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize for faster and more robust contour detection
        orig = image.copy()
        ratio = image.shape[0] / 500.0
        res = cv2.resize(image, (int(image.shape[1]/ratio), 500))
        gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold to isolate the bright receipt from background
        blurred = cv2.GaussianBlur(gray_res, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological closing to fill gaps in the receipt
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = res.shape[0] * res.shape[1]
        
        # Filter out contours that are too small or too large (e.g. the whole image border)
        valid = [c for c in contours if img_area * 0.1 < cv2.contourArea(c) < img_area * 0.95]
        
        if not valid:
            # Fallback to Canny if thresholding fails to find a good contour
            edged = cv2.Canny(blurred, 50, 150)
            edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if img_area * 0.1 < cv2.contourArea(c) < img_area * 0.95]
            
            if not valid:
                return image # Still failed, return original
                
        largest_cnt = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        # Scale back to original dimensions
        x, y, w, h = int(x*ratio), int(y*ratio), int(w*ratio), int(h*ratio)
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        return image[y:y+h, x:x+w]

    @staticmethod
    def process_all(image: np.ndarray) -> np.ndarray:
        # Step 0: Crop out the background
        cropped = ImagePreprocessor.crop_document(image)
        # Step 1: Deskew
        deskewed = ImagePreprocessor.deskew(cropped)
        # Step 2-5: Enhance
        enhanced = ImagePreprocessor.enhance(deskewed)
        return enhanced


class TextPreprocessor:
    """
    Handles text and bounding box preprocessing (e.g. from OCR output) before sending to KIE models.
    """
    
    @staticmethod
    def sort_reading_order(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts bounding boxes into a perfect reading order (top-to-bottom, left-to-right).
        Expects a list of dictionaries, each containing a 'box' key [x1, y1, x2, y2].
        """
        if not boxes:
            return []
            
        # Calculate centers and heights
        enriched_boxes = []
        for item in boxes:
            x1, y1, x2, y2 = item['box']
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            h = y2 - y1
            enriched_boxes.append({
                'original': item,
                'cx': cx,
                'cy': cy,
                'h': h,
                'x1': x1,
                'y1': y1
            })
            
        # Sort by Y-center initially
        enriched_boxes.sort(key=lambda x: x['cy'])
        
        # Calculate median height to use as threshold
        if len(enriched_boxes) > 0:
            median_h = np.median([b['h'] for b in enriched_boxes])
        else:
            median_h = 10
            
        y_threshold = median_h * 0.5 # Two boxes are on the same line if their Y-centers differ by less than half a character height
        
        lines = []
        current_line = []
        
        for item in enriched_boxes:
            if not current_line:
                current_line.append(item)
            else:
                # Compare with the average cy of the current line
                avg_cy = sum(b['cy'] for b in current_line) / len(current_line)
                if abs(item['cy'] - avg_cy) <= y_threshold:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
        
        if current_line:
            lines.append(current_line)
            
        # Sort within each line by X-center (Left to Right)
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda x: x['x1'])
            for item in line:
                sorted_boxes.append(item['original'])
                
        return sorted_boxes

# Simple test if run directly
if __name__ == "__main__":
    print("Preprocessing module loaded.")
