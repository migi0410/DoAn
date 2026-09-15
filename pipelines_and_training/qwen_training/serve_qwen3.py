import os
import sys
import json
import time
import io
import re
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

app = FastAPI(title="Qwen3-VL LoRA v2 Enhanced Serving API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR_V2 = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora_v2/final_lora_checkpoint"
MODEL_DIR_V1 = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora/final_lora_checkpoint"
model = None
processor = None
active_version = "v2"

# CHUẨN 100% PROMPT V2 ENHANCED ĐÃ HUẤN LUYỆN
PROMPT_SCHEMA_V2 = """Bạn là chuyên gia trích xuất thông tin hóa đơn tiếng Việt. Hãy đọc kỹ hình ảnh và trích xuất thông tin vào định dạng JSON sau:
{
  "SELLER": "Tên cửa hàng hoặc công ty",
  "ADDRESS": "Địa chỉ",
  "TIMESTAMP": "Thời gian lập hóa đơn",
  "ITEMS": [
    {
      "name": "Tên món hàng",
      "qty": "Số lượng",
      "price": "Đơn giá",
      "amount": "Thành tiền"
    }
  ],
  "TOTAL_COST": "Tổng tiền thanh toán cuối cùng"
}
Quy tắc bắt buộc:
1. Chỉ trích xuất CHÍNH XÁC những gì nhìn thấy trên ảnh. Giữ nguyên dấu tiếng Việt và chính tả gốc trên hóa đơn. Tuyệt đối KHÔNG tự ý thêm/bớt dấu và KHÔNG bịa đặt thông tin.
2. Với mỗi mặt hàng, nhóm đủ 4 trường: name, qty, price, amount vào cùng một đối tượng.
3. Nếu tên một món hàng dài bị in rớt xuống 2-3 dòng, hãy ghép lại thành một tên món hoàn chỉnh duy nhất.
4. Dòng in số lượng x đơn giá (ví dụ: '1.000 KG x 11.900') hoặc dòng khuyến mãi/giảm giá nằm ngay bên dưới tên món phải được gộp đúng vào món hàng đó, KHÔNG tách thành các món rời rạc.
5. Nếu hóa đơn không in đơn giá riêng (chỉ có số lượng và thành tiền), hãy để price là "".
6. Nếu trường thông tin nào không xuất hiện trên hóa đơn, hãy để giá trị là ""."""

@app.on_event("startup")
def load_model():
    global model, processor, active_version
    print("🚀 Đang nạp model Qwen3-VL 4-bit với Enhanced LoRA v2...")
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if os.path.exists(MODEL_DIR_V2):
        print(f"Ghép trọng số LoRA v2 Enhanced từ: {MODEL_DIR_V2}")
        model = PeftModel.from_pretrained(base_model, MODEL_DIR_V2)
        active_version = "v2"
    elif os.path.exists(MODEL_DIR_V1):
        print(f"Ghép trọng số LoRA v1 từ: {MODEL_DIR_V1}")
        model = PeftModel.from_pretrained(base_model, MODEL_DIR_V1)
        active_version = "v1"
    else:
        print("Dùng base model...")
        model = base_model
        active_version = "base"
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256*28*28,
        max_pixels=1024*28*28,
        trust_remote_code=True
    )
    print(f"✅ Model Qwen3-VL LoRA ({active_version} Enhanced) đã sẵn sàng phục vụ API!")

@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "active_version": active_version,
        "model_dir": MODEL_DIR_V2 if active_version == "v2" else MODEL_DIR_V1
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    t0 = time.time()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    w, h = image.size
    max_dim = 1536
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_SCHEMA_V2}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
    out_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n" + "="*50)
    print("📥 INCOMING IMAGE SIZE:", (w, h))
    print("📤 RAW GENERATED OUTPUT:")
    print(out_text)
    print("="*50 + "\n", flush=True)
    
    # Parse JSON
    parsed = {}
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', out_text, re.DOTALL)
    if m:
        try: parsed = json.loads(m.group(1))
        except: pass
    if not parsed:
        m2 = re.search(r'\{.*\}', out_text, re.DOTALL)
        if m2:
            try: parsed = json.loads(m2.group(0))
            except: pass

    # Backward compatibility
    if "ITEMS" in parsed and isinstance(parsed["ITEMS"], list):
        if "ITEM_NAME" not in parsed:
            parsed["ITEM_NAME"] = [item.get("name", "") for item in parsed["ITEMS"]]
            parsed["ITEM_QTY"] = [item.get("qty", "") for item in parsed["ITEMS"]]
            parsed["ITEM_PRICE"] = [item.get("price", "") for item in parsed["ITEMS"]]
            parsed["ITEM_AMOUNT"] = [item.get("amount", "") for item in parsed["ITEMS"]]
            
    latency = round(time.time() - t0, 2)
    return {"success": True, "data": parsed, "raw": out_text, "latency_s": latency, "model_version": active_version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
