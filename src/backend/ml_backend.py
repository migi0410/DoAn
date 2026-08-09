import os
import io
import json
import random
import string
import sys
import urllib.request
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from paddleocr import PaddleOCR

\
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="AVIR-KIE PaddleOCR Label Studio ML Backend")

ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        print("Initializing PaddleOCR model (Vietnamese language)...")
        ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4")
        print("PaddleOCR model initialized successfully!")
    return ocr

def generate_random_id():
    """Generates a random 10-character alphanumeric ID for Label Studio linking."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=10))

def load_image(image_path_or_url):
    """Downloads image if it's a URL, or opens it from local disk."""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        \
        url = image_path_or_url
        if "localhost" in url:
            \
            url = url.replace("localhost", "127.0.0.1")
            
        print(f"Downloading image from URL: {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            img_bytes = response.read()
        return Image.open(io.BytesIO(img_bytes))
    else:
        \
        print(f"Loading local image path: {image_path_or_url}")
        filename = os.path.basename(image_path_or_url)
        
        \
        possible_paths = [
            image_path_or_url,
            os.path.join("/app/data/media/upload", filename),
            os.path.join(os.path.expanduser("~"), ".local/share/label-studio/media/upload", filename)
        ]
        
        \
        if image_path_or_url.startswith("/data/upload/"):
            possible_paths.append(image_path_or_url.replace("/data/upload/", "/app/data/media/upload/"))
            
        media_dir = "/app/data/media/upload"
        if os.path.exists(media_dir):
            for root, dirs, files in os.walk(media_dir):
                if filename in files:
                    possible_paths.append(os.path.join(root, filename))
                    break
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found image at: {path}")
                return Image.open(path)
                
        raise FileNotFoundError(f"Could not load image from: {image_path_or_url}")

def heuristic_classify(text, y_pct):
    """Simple heuristic to pre-classify OCR text into KIE labels."""
    text_upper = text.upper()
    
    \
    if any(k in text_upper for k in ["NGÀY", "GIỜ", "THỜI GIAN", "DATE", "TIME"]):
        return "TIMESTAMP"
    if any(c.isdigit() for c in text_upper) and ("/" in text_upper or ":" in text_upper):
        \
        return "TIMESTAMP"
        
    if any(k in text_upper for k in ["TỔNG TIỀN", "TỔNG CỘNG", "THANH TOÁN", "CỘNG TIỀN", "TOTAL COST", "GRAND TOTAL"]):
        return "TOTAL_COST"
        
    if y_pct < 18:
        if any(k in text_upper for k in ["WINMART", "GS25", "COOPMART", "CO.OPMART", "PHARMACITY", "COFFEE", "CAFE", "SIÊU THỊ", "CỬA HÀNG", "NHÀ THUỐC"]):
            return "SELLER"
            
    if any(k in text_upper for k in ["ĐỊA CHỈ", "Đ/C", "ĐC:", "ADDRESS", "QUẬN", "PHƯỜNG", "ĐƯỜNG", "THÀNH PHỐ", "TP.", "TỈNH"]):
        return "ADDRESS"
        
    return "OTHER"

@app.get("/health")
@app.get("/")
def health():
    """Health check endpoint required by Label Studio."""
    return {"status": "UP"}

@app.post("/setup")
def setup(request: Request):
    """Setup endpoint returning model information."""
    return {"model_version": "PaddleOCR-v4-VI"}

@app.post("/predict")
async def predict(request: Request):
    """
    Main prediction endpoint called by Label Studio.
    Processes tasks, runs OCR, converts coordinates to percentages,
    and returns linked pre-annotations (rectangle + text).
    """
    body = await request.json()
    print("Received predict request:")
    print(json.dumps(body, indent=2))
    
    tasks = body.get("tasks", [])
    predictions = []
    
    \
    FROM_NAME_RECT = "label"
    FROM_NAME_TEXT = "transcription"
    TO_NAME = "image"
    
    for task in tasks:
        task_id = task.get("id")
        image_data = task.get("data", {})
        
        \
        image_path = image_data.get("image", image_data.get("ocr", ""))
        if not image_path:
            continue
            
        try:
            \
            img = load_image(image_path)
            W, H = img.size
            print(f"Loaded image size: {W}x{H}")
            
            \
\
\
            temp_img_path = f"temp_ocr_{task_id}.png"
            img.save(temp_img_path)
            
            ocr_result = get_ocr().ocr(temp_img_path)
            
            \
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
            results = []
            
            \
\
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    box = line[0]\
                    text_info = line[1]\
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    \
                    xmin = min(p[0] for p in box)
                    ymin = min(p[1] for p in box)
                    xmax = max(p[0] for p in box)
                    ymax = max(p[1] for p in box)
                    
                    \
                    xmin = max(0.0, min(xmin, float(W)))
                    ymin = max(0.0, min(ymin, float(H)))
                    xmax = max(0.0, min(xmax, float(W)))
                    ymax = max(0.0, min(ymax, float(H)))
                    
                    \
                    x_pct = (xmin / W) * 100
                    y_pct = (ymin / H) * 100
                    w_pct = ((xmax - xmin) / W) * 100
                    h_pct = ((ymax - ymin) / H) * 100
                    
                    \
                    kie_label = heuristic_classify(text, y_pct)
                    
                    \
                    box_id = generate_random_id()
                    
                    \
                    results.append({
                        "id": box_id,
                        "from_name": FROM_NAME_RECT,
                        "to_name": TO_NAME,
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_pct,
                            "y": y_pct,
                            "width": w_pct,
                            "height": h_pct,
                            "rectanglelabels": [kie_label]
                        },
                        "score": float(confidence)
                    })
                    
                    \
                    results.append({
                        "id": box_id,
                        "from_name": FROM_NAME_TEXT,
                        "to_name": TO_NAME,
                        "type": "textarea",
                        "value": {
                            "x": x_pct,
                            "y": y_pct,
                            "width": w_pct,
                            "height": h_pct,
                            "text": [text]
                        },
                        "score": float(confidence)
                    })
            
            predictions.append({
                "result": results,
                "score": 0.95\
            })
            
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            \
            predictions.append({
                "result": [],
                "score": 0.0,
                "error": str(e)
            })
            
    return JSONResponse(content={"results": predictions})

if __name__ == "__main__":
    import uvicorn
    \
    uvicorn.run(app, host="0.0.0.0", port=9090)
