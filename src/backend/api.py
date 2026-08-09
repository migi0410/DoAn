import os
import sys
# Thêm đường dẫn tuyệt đối của thư mục chứa api.py vào sys.path để import cục bộ không bị lỗi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import uuid
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil

app = FastAPI(title="AVIR-KIE Inference API")

\
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],\
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
        \
        file_id = str(uuid.uuid4())[:8]
        ext = file.filename.split('.')[-1]
        img_path = os.path.join(UPLOAD_DIR, f"upload_{file_id}.{ext}")
        
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        preprocessed_url = None
        \
        if preprocess:
            try:
                from utils.preprocessing import ImagePreprocessor
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    processed = ImagePreprocessor.process_all(img)
                    preprocessed_path = os.path.join(UPLOAD_DIR, f"preprocessed_{file_id}.{ext}")
                    cv2.imwrite(preprocessed_path, processed)
                    img_path = preprocessed_path
                    preprocessed_url = f"/api/image/{os.path.basename(preprocessed_path)}"
            except Exception as e:
                print("CV Preprocessing error:", e)
            
        result, words, bboxes = registry.predict(baseline, img_path, preprocess=preprocess)
        print(f"===== FINAL RESULT TO FRONTEND =====\n{result}\n=====================================")

        # Draw bboxes on image
        result_filename = f"result_{file_id}.{ext}"
        result_path = os.path.join(UPLOAD_DIR, result_filename)
        try:
            import cv2, numpy as np
            img = cv2.imread(img_path)
            if img is not None and bboxes:
                for box in bboxes:
                    try:
                        pts = np.array([[int(p[0]), int(p[1])] for p in box], dtype=np.int32)
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    except Exception:
                        continue
                cv2.imwrite(result_path, img)
            else:
                shutil.copy(img_path, result_path)
        except Exception as e:
            print("BBox drawing error:", e)
            shutil.copy(img_path, result_path)

        return JSONResponse(content={
            "success": True,
            "extraction": result,
            "image_url": f"/api/image/{result_filename}",
            "preprocessed_url": preprocessed_url
        })
        
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        print("API ERROR:", err_str)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": err_str})

@app.post("/api/chat")
async def chat(
    file: UploadFile = File(...),
    model: str = Form(...),
    question: str = Form(...),
):
    """VLM Q&A: Ask anything about the invoice image."""
    try:
        file_id = str(uuid.uuid4())[:8]
        ext = file.filename.split('.')[-1]
        img_path = os.path.join(UPLOAD_DIR, f"chat_{file_id}.{ext}")
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        answer = registry.chat(model, img_path, question)
        return JSONResponse(content={"success": True, "answer": answer})

    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        print("API ERROR:", err_str)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": err_str})


@app.get("/api/image/{filename}")
def get_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Image not found"})
