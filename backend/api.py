import os
import sys
import json
import uuid
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil

app = FastAPI(title="AVIR-KIE Inference API")

# Setup CORS for Frontend (Next.js runs on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*"], # Replace https://* with Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AVIR-KIE API is running"}

from inference_engine import ModelRegistry

registry = None

@app.on_event("startup")
async def startup_event():
    global registry
    registry = ModelRegistry()

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    baseline: str = Form(...),
    preprocess: bool = Form(False)
):
    try:
        # 1. Save uploaded file
        file_id = str(uuid.uuid4())[:8]
        ext = file.filename.split('.')[-1]
        img_path = os.path.join(UPLOAD_DIR, f"upload_{file_id}.{ext}")
        
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        preprocessed_url = None
        # Optional: Preprocess image with OpenCV
        if preprocess:
            try:
                import sys
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
                from utils.preprocessing import ImagePreprocessor
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    processed = ImagePreprocessor.process_all(img)
                    preprocessed_path = os.path.join(UPLOAD_DIR, f"preprocessed_{file_id}.{ext}")
                    cv2.imwrite(preprocessed_path, processed)
                    img_path = preprocessed_path # override to use preprocessed image
                    preprocessed_url = f"/api/image/{os.path.basename(preprocessed_path)}"
            except Exception as e:
                print("CV Preprocessing error:", e)
            
        # 2. Run Inference via Registry
        result, words, bboxes = registry.predict(baseline, img_path, preprocess=preprocess)
        
        # 3. Draw Bounding Boxes
        import cv2
        img = cv2.imread(img_path)
        for box in bboxes:
            box = [(int(p[0]), int(p[1])) for p in box]
            cv2.polylines(img, [__import__("numpy").array(box)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        result_filename = f"result_{file_id}.{ext}"
        result_path = os.path.join(UPLOAD_DIR, result_filename)
        cv2.imwrite(result_path, img)
        
        return JSONResponse(content={
            "success": True,
            "extraction": result,
            "image_url": f"/api/image/{result_filename}",
            "preprocessed_url": preprocessed_url
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/image/{filename}")
def get_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Image not found"})
