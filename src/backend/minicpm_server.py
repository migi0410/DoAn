"""
Standalone MiniCPM-V inference server.
Runs on port 8001 with its own Python venv (transformers==4.40).
Start via: /workspace/minicpm_env/bin/python3 minicpm_server.py
"""
import os, json, re, uuid, shutil, builtins, typing

# Inject typing into builtins for modeling_minicpmv.py compatibility
for name in ["List", "Optional", "Dict", "Any", "Tuple", "Union", "Set", "Callable"]:
    if not hasattr(builtins, name):
        setattr(builtins, name, getattr(typing, name))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="MiniCPM-V Server")

MODEL_DIR    = os.environ.get("MINICPM_MODEL_DIR", "/workspace/minicpm_v_lora_official")
# Use the official int4 model to bypass dynamic quantization bugs
BASE_MODEL = "openbmb/MiniCPM-V-2_6-int4"
UPLOAD_DIR   = "/tmp/minicpm_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_model     = None
_tokenizer = None

@app.on_event("startup")
async def startup():
    global _model, _tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[MiniCPM-Server] Loading base: {BASE_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    _model = AutoModel.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if os.path.exists(MODEL_DIR):
        print(f"[MiniCPM-Server] Loading LoRA: {MODEL_DIR}")
        try:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, MODEL_DIR)
        except Exception as e:
            print(f"[MiniCPM-Server] LoRA failed ({e}), using base model.")
    _model.eval()
    print("[MiniCPM-Server] Ready!")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    try:
        from PIL import Image
        ext = file.filename.split(".")[-1] if "." in (file.filename or "") else "jpg"
        tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image = Image.open(tmp).convert("RGB")
        prompt = ("Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, "
                  "ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON.")
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = _model.chat(image=None, msgs=msgs, tokenizer=_tokenizer,
                          sampling=False, max_new_tokens=2048)
        os.unlink(tmp)
        result = _parse_vlm_response(res)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

def _parse_vlm_response(response: str) -> dict:
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        start = response.find("{")
        end   = response.rfind("}")
        if start != -1 and end > start:
            data = json.loads(response[start:end+1])
        else:
            data = json.loads(response)
        data_upper = {k.upper(): v for k, v in data.items()}

        def unpack(v):
            if isinstance(v, list) and v: return str(v[0])
            return str(v) if v is not None else ""

        item_keys = ["ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]
        for k in item_keys:
            if k in data_upper and not isinstance(data_upper[k], list):
                data_upper[k] = [data_upper[k]] if data_upper[k] else []

        max_len = max((len(data_upper.get(k, [])) for k in item_keys), default=0)
        items = []
        for i in range(max_len):
            item = {k: unpack(data_upper.get(k, [])[i]) if i < len(data_upper.get(k, [])) else "" for k in item_keys}
            items.append(item)

        clean = {k: unpack(v) for k, v in data_upper.items() if k not in item_keys and k != "ITEMS"}
        if items:
            clean["ITEMS"] = items
        return clean
    except Exception as e:
        return {"OTHER": response, "parse_error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
