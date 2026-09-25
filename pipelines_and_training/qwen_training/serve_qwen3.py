import os
import sys
import json
import time
import io
import re
import gc
import base64
import requests
import torch
from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

app = FastAPI(title="AVIR-KIE Multi-Model Universal Serving API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID_BASE = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR_V2 = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora_v2/final_lora_checkpoint"
MODEL_DIR_V1 = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora/final_lora_checkpoint"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global state
base_model = None
model = None
processor = None
active_adapter = "v2"
available_adapters = []
is_loading = False

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
1. NGUYÊN TẮC GIÓNG HÀNG THEO CỘT THÀNH TIỀN: Mỗi phần tử trong mảng ITEMS bắt buộc phải tương ứng với đúng MỘT dòng in thành tiền ở cột Thành tiền (T. Tiền). Tuyệt đối không lặp lại cùng một số tiền cho 2 phần tử khác nhau.
2. NGUYÊN TẮC GHÉP DÒNG: Nếu tên của một món hàng dài bị in rớt xuống 2 hoặc 3 dòng (ví dụ: 'NAM DƯƠNG Sốt' xuống dòng 'Dầu Dấm Trộn' xuống dòng 'Salad 250g'; hoặc 'WINECO Xà lách' xuống dòng 'lolo xanh L1 300g') mà các dòng dưới KHÔNG có in số tiền mới ở cột Thành tiền, BẮT BUỘC phải ghép tất cả các dòng đó lại thành MỘT tên món duy nhất ('NAM DƯƠNG Sốt Dầu Dấm Trộn Salad 250g', 'WINECO Xà lách lolo xanh L1 300g') ứng với đúng số tiền thực tế của món đó.
3. TUYỆT ĐỐI KHÔNG tách các dòng tiếp nối thành các món riêng và TUYỆT ĐỐI KHÔNG bốc số tiền của món bên dưới gán cho món bên trên.
4. Dòng khuyến mãi / giảm giá (như 'KM: -3,100') nằm dưới món hàng phải gộp vào món hàng đó, không tạo thêm món.
5. QUY TẮC NGUYÊN VĂN (VERBATIM): Sao chép chính xác 100% từng ký tự như in trên ảnh hóa đơn. Nếu chữ in trên hóa đơn là KHÔNG DẤU (ví dụ: 'SUA TUOI', 'CA TIM', 'XUONG HEO') thì BẮT BUỘC giữ nguyên KHÔNG DẤU ('SUA TUOI', 'CA TIM', 'XUONG HEO'). TUYỆT ĐỐI KHÔNG tự ý suy đoán, không tự thêm dấu tiếng Việt và không tự sửa chính tả."""

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
        reserved = round(torch.cuda.memory_reserved() / (1024 ** 2), 1)
        return {"allocated_mb": allocated, "reserved_mb": reserved}
    return {"allocated_mb": 0.0, "reserved_mb": 0.0}

def kick_model_from_vram():
    """Giải phóng triệt để 100% VRAM của mô hình PyTorch hiện tại."""
    global model, base_model, processor, active_adapter, available_adapters
    print("🧹 [VRAM Purge] Bắt đầu kick mô hình ra khỏi VRAM...")
    
    if model is not None:
        try:
            model.to("cpu")
        except Exception as e:
            print(f"Warning moving model to cpu: {e}")
        del model
        model = None

    if base_model is not None:
        try:
            base_model.to("cpu")
        except Exception as e:
            print(f"Warning moving base_model to cpu: {e}")
        del base_model
        base_model = None

    if processor is not None:
        del processor
        processor = None

    active_adapter = "unloaded"
    available_adapters = []

    # Thu hồi rác và xả cache CUDA
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    # Cũng gửi lệnh unload cho Ollama nếu có model đang treo
    try:
        requests.post(OLLAMA_URL, json={"model": "minicpm-v:latest", "keep_alive": 0}, timeout=2)
        requests.post(OLLAMA_URL, json={"model": "deepseek-ocr:latest", "keep_alive": 0}, timeout=2)
        requests.post(OLLAMA_URL, json={"model": "qwen2.5:7b", "keep_alive": 0}, timeout=2)
    except Exception:
        pass

    vram = get_vram_usage()
    print(f"✅ [VRAM Purge Hoàn Tất] VRAM còn lại: {vram['allocated_mb']}MB allocated, {vram['reserved_mb']}MB reserved.")
    return vram

def load_qwen3_multi_adapter():
    """Nạp Base Model 4-bit và nạp đồng thời các Adapter để hỗ trợ Hot-Swapping 0ms."""
    global model, base_model, processor, active_adapter, available_adapters, is_loading
    if model is not None:
        return model, processor

    is_loading = True
    print("🚀 [VRAM Load] Đang nạp Base Model Qwen3-VL 4-bit...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    adapters = []
    # 1. Nạp LoRA v2 làm adapter chính
    if os.path.exists(MODEL_DIR_V2):
        print(f"📦 Ghép LoRA v2 Enhanced từ: {MODEL_DIR_V2}")
        peft_m = PeftModel.from_pretrained(base, MODEL_DIR_V2, adapter_name="v2")
        adapters.append("v2")
        
        # 2. Nạp thêm LoRA v1 vào cùng mô hình
        if os.path.exists(MODEL_DIR_V1):
            try:
                print(f"📦 Ghép bổ sung LoRA v1 từ: {MODEL_DIR_V1}")
                peft_m.load_adapter(MODEL_DIR_V1, adapter_name="v1")
                adapters.append("v1")
            except Exception as e:
                print(f"Không thể nạp v1 adapter: {e}")
        
        adapters.append("base") # Base model zero-shot
        peft_m.set_adapter("v2")
        model = peft_m
        active_adapter = "v2"
    elif os.path.exists(MODEL_DIR_V1):
        print(f"📦 Ghép LoRA v1 từ: {MODEL_DIR_V1}")
        peft_m = PeftModel.from_pretrained(base, MODEL_DIR_V1, adapter_name="v1")
        adapters.append("v1")
        adapters.append("base")
        peft_m.set_adapter("v1")
        model = peft_m
        active_adapter = "v1"
    else:
        print("📦 Chạy trực tiếp Base Model (không dùng LoRA)...")
        model = base
        adapters.append("base")
        active_adapter = "base"

    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID_BASE,
        min_pixels=256*28*28,
        max_pixels=1024*28*28,
        trust_remote_code=True
    )

    available_adapters = adapters
    is_loading = False
    vram = get_vram_usage()
    print(f"✅ Nạp hoàn tất trong {round(time.time() - t0, 2)}s! VRAM: {vram['allocated_mb']}MB. Adapters có sẵn: {available_adapters}")
    return model, processor

def parse_vlm_output(out_text: str) -> dict:
    """Bóc tách JSON chuẩn từ chuỗi sinh ra của các mô hình VLM."""
    parsed = {}
    if not out_text:
        return parsed
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', out_text, re.DOTALL)
    if m:
        try: parsed = json.loads(m.group(1))
        except: pass
    if not parsed:
        m2 = re.search(r'\{.*\}', out_text, re.DOTALL)
        if m2:
            try: parsed = json.loads(m2.group(0))
            except: pass

    # Chuẩn hóa cấu trúc nếu có lồng dict phức tạp (như của MiniCPM)
    if isinstance(parsed.get("SELLER"), dict):
        parsed["SELLER"] = parsed["SELLER"].get("name", "")
    if isinstance(parsed.get("TOTAL_COST"), dict):
        parsed["TOTAL_COST"] = str(parsed["TOTAL_COST"].get("value", ""))
    if isinstance(parsed.get("ADDRESS"), list):
        parsed["ADDRESS"] = ", ".join(str(x) for x in parsed["ADDRESS"])
    elif isinstance(parsed.get("ADDRESS"), dict):
        parsed["ADDRESS"] = " ".join(str(v) for v in parsed["ADDRESS"].values())

    # Chuẩn hóa ITEMS
    if "ITEMS" in parsed and isinstance(parsed["ITEMS"], list):
        clean_items = []
        for it in parsed["ITEMS"]:
            if isinstance(it, dict):
                clean_items.append({
                    "name": str(it.get("name", "")),
                    "qty": str(it.get("qty", "1")),
                    "price": str(it.get("price", "")),
                    "amount": str(it.get("amount", ""))
                })
        parsed["ITEMS"] = clean_items
        if "ITEM_NAME" not in parsed:
            parsed["ITEM_NAME"] = [it["name"] for it in clean_items]
            parsed["ITEM_QTY"] = [it["qty"] for it in clean_items]
            parsed["ITEM_PRICE"] = [it["price"] for it in clean_items]
            parsed["ITEM_AMOUNT"] = [it["amount"] for it in clean_items]

    return parsed

def parse_kie_regex(ocr_text: str) -> dict:
    """Bộ bóc tách luật Heuristic Regex truyền thống trên văn bản OCR thô."""
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    seller = lines[0] if lines else "Cửa hàng"
    address = ""
    timestamp = ""
    total = ""
    
    # 1. Tìm địa chỉ bằng từ khóa
    addr_kw = ["đ/c", "địa chỉ", "tầng", "số", "phường", "quận", "đường", "street", "tang", "tp.", "hcm", "hà nội"]
    for line in lines:
        if any(kw in line.lower() for kw in addr_kw) and not address:
            address = re.sub(r"^(đ/c|địa chỉ)[:\s\-]*", "", line, flags=re.I).strip()
            
    # 2. Tìm ngày giờ bằng regex
    date_p = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    for line in lines:
        m = re.search(date_p, line)
        if m and not timestamp:
            timestamp = m.group(1)

    # 3. Tìm tổng tiền
    cost_kw = ["tổng tiền", "tổng cộng", "thanh toán", "cần thanh toán", "total", "t.tien", "thành tiền"]
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in cost_kw) and not total:
            nums = re.findall(r"\d{1,3}(?:[.,]\d{3})+", line)
            if nums:
                total = nums[-1]
            elif i + 1 < len(lines):
                next_nums = re.findall(r"\d{1,3}(?:[.,]\d{3})+", lines[i+1])
                if next_nums:
                    total = next_nums[-1]

    # 4. Trích xuất các món theo luật regex (dễ bị sót hoặc dính số)
    items = []
    for line in lines:
        nums = re.findall(r"\d{1,3}(?:[.,]\d{3})+", line)
        if nums and not any(kw in line.lower() for kw in ["tổng", "thanh toán", "mst", "ngày", "date", "cash", "tiền mặt", "quay"]):
            name = re.sub(r"\d+.*$", "", line).strip()
            if len(name) >= 3:
                amount = nums[-1]
                price = nums[-2] if len(nums) >= 2 else ""
                items.append({
                    "name": name,
                    "qty": "1",
                    "price": price,
                    "amount": amount
                })

    return {
        "SELLER": seller,
        "ADDRESS": address,
        "TIMESTAMP": timestamp,
        "TOTAL_COST": total,
        "ITEMS": items[:6],
        "ITEM_NAME": [it["name"] for it in items[:6]],
        "ITEM_QTY": [it["qty"] for it in items[:6]],
        "ITEM_PRICE": [it["price"] for it in items[:6]],
        "ITEM_AMOUNT": [it["amount"] for it in items[:6]]
    }

@app.on_event("startup")
def startup_event():
    load_qwen3_multi_adapter()

@app.get("/health")
def health():
    vram = get_vram_usage()
    return {
        "status": "loading" if is_loading else ("ready" if model is not None else "unloaded"),
        "active_adapter": active_adapter,
        "available_adapters": available_adapters,
        "vram_allocated_mb": vram["allocated_mb"],
        "vram_reserved_mb": vram["reserved_mb"],
        "active_version": active_adapter
    }

@app.post("/unload")
def unload_vram():
    """API chủ động kick toàn bộ mô hình ra khỏi VRAM GPU."""
    vram = kick_model_from_vram()
    return {
        "success": True,
        "message": "Đã giải phóng sạch mô hình khỏi VRAM GPU!",
        "vram": vram
    }

@app.post("/switch_adapter")
def switch_adapter_endpoint(target: str = Form(...)):
    """Chuyển đổi Adapter tức thì (0ms overhead). Target: 'v2', 'v1', hoặc 'base'."""
    global model, active_adapter
    if model is None:
        load_qwen3_multi_adapter()

    target_clean = target.strip().lower()
    if target_clean in ["qwen3_lora_v2", "v2", "lora_v2"]:
        target_name = "v2"
    elif target_clean in ["qwen3_lora_v1", "v1", "lora_v1"]:
        target_name = "v1"
    elif target_clean in ["qwen3_base_v2", "base", "zero_shot"]:
        target_name = "base"
    else:
        target_name = target_clean

    if target_name in ["base", "zero_shot"]:
        active_adapter = "base"
        return {"success": True, "active_adapter": "base", "message": "Đã chuyển sang Base Model (Tắt LoRA). Thời gian: 0ms"}
    elif target_name in available_adapters and isinstance(model, PeftModel):
        try:
            model.set_adapter(target_name)
            active_adapter = target_name
            return {"success": True, "active_adapter": target_name, "message": f"Đã chuyển sang LoRA {target_name}. Thời gian: 0ms"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi set_adapter: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Adapter '{target_name}' không có sẵn. Danh sách: {available_adapters}")

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form("qwen3_lora_v2"),
    custom_prompt: Optional[str] = Form(None)
):
    global model, processor, active_adapter
    t0 = time.time()
    contents = await file.read()
    mid = (model_id or "qwen3_lora_v2").lower()

    # =========================================================================
    # BRANCH 1: MINICPM-V 2.6 (8B) - THỰC THI TRỰC TIẾP TRÊN GPU QUA OLLAMA
    # =========================================================================
    if "minicpm" in mid:
        print(f"🚀 [MiniCPM-V Execution] Đang chạy mô hình MiniCPM-V 2.6 (8B) trên GPU...")
        try:
            img_b64 = base64.b64encode(contents).decode("utf-8")
            payload = {
                "model": "minicpm-v:latest",
                "prompt": PROMPT_SCHEMA_V2,
                "stream": False,
                "images": [img_b64],
                "keep_alive": 0,  # KICK VRAM ngay sau khi sinh xong!
                "options": {"temperature": 0.0, "num_predict": 1024}
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                out_text = resp.json().get("response", "")
                parsed = parse_vlm_output(out_text)
                latency = round(time.time() - t0, 2)
                print(f"✅ [MiniCPM-V Thành Công] Thời gian: {latency}s. Số món: {len(parsed.get('ITEMS', []))}")
                return {
                    "success": True,
                    "data": parsed,
                    "raw": out_text,
                    "latency_s": latency,
                    "model_version": "minicpm_v",
                    "vram": get_vram_usage()
                }
        except Exception as e:
            print(f"⚠️ Lỗi khi chạy MiniCPM-V qua Ollama: {e}")

    # =========================================================================
    # BRANCH 2: DEEPSEEK-OCR + HEURISTIC REGEX / QWEN2.5 - THỰC THI TRÊN GPU
    # =========================================================================
    if "deepseek" in mid or "regex" in mid:
        print(f"🚀 [DeepSeek Pipeline] Đang chạy DeepSeek-OCR trên GPU...")
        try:
            img_b64 = base64.b64encode(contents).decode("utf-8")
            # Bước 1: Nhận diện văn bản OCR thô
            payload_ocr = {
                "model": "deepseek-ocr:latest",
                "prompt": "Read all text from this receipt:",
                "stream": False,
                "images": [img_b64],
                "keep_alive": 0,  # KICK VRAM ngay sau khi OCR xong!
                "options": {"num_predict": 1024}
            }
            resp_ocr = requests.post(OLLAMA_URL, json=payload_ocr, timeout=120)
            ocr_text = resp_ocr.json().get("response", "") if resp_ocr.status_code == 200 else ""
            
            if "regex" in mid:
                # Bước 2a: Dùng Heuristic Regex truyền thống bóc tách
                parsed = parse_kie_regex(ocr_text)
                current_active = "deepseek_regex"
                out_text = ocr_text
            else:
                # Bước 2b: Đẩy OCR text vào Qwen2.5 (7B) để trích xuất JSON
                payload_qwen = {
                    "model": "qwen2.5:7b",
                    "prompt": f"Dưới đây là văn bản OCR từ hóa đơn:\n{ocr_text}\nHãy trích xuất thông tin sang JSON theo định dạng sau:\n{PROMPT_SCHEMA_V2}",
                    "stream": False,
                    "keep_alive": 0,  # KICK VRAM ngay sau khi xong!
                    "options": {"temperature": 0.0, "num_predict": 1024}
                }
                resp_qwen = requests.post(OLLAMA_URL, json=payload_qwen, timeout=90)
                out_text = resp_qwen.json().get("response", "") if resp_qwen.status_code == 200 else ""
                parsed = parse_vlm_output(out_text)
                current_active = "deepseek_qwen25"

            latency = round(time.time() - t0, 2)
            print(f"✅ [DeepSeek Thành Công] Model: {current_active}, Thời gian: {latency}s. Số món: {len(parsed.get('ITEMS', []))}")
            return {
                "success": True,
                "data": parsed,
                "raw": out_text,
                "latency_s": latency,
                "model_version": current_active,
                "vram": get_vram_usage()
            }
        except Exception as e:
            print(f"⚠️ Lỗi khi chạy DeepSeek pipeline: {e}")

    # =========================================================================
    # BRANCH 3: QWEN3-VL LO-RA V2 / BASE / LO-RA V1 - PYTORCH PEFT TRÊN GPU
    # =========================================================================
    if model is None or processor is None:
        print("⚠️ Qwen3-VL đang bị kick, tự động nạp lại vào VRAM...")
        load_qwen3_multi_adapter()

    is_base = "base" in mid
    is_v1 = "v1" in mid
    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = image.size
    max_dim = 1536
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": custom_prompt if custom_prompt else PROMPT_SCHEMA_V2}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    # Thực thi với Adapter tương ứng (Hot-Swapping 0ms)
    with torch.no_grad():
        if isinstance(model, PeftModel):
            if is_base or active_adapter == "base":
                with model.disable_adapter():
                    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                    current_active = "base"
            elif is_v1 and "v1" in available_adapters:
                model.set_adapter("v1")
                outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                current_active = "v1"
                active_adapter = "v1"
            else:
                if "v2" in available_adapters:
                    model.set_adapter("v2")
                    current_active = "v2"
                    active_adapter = "v2"
                else:
                    current_active = active_adapter
                outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        else:
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            current_active = "base"
        
    out_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    parsed = parse_vlm_output(out_text)
    latency = round(time.time() - t0, 2)
    vram = get_vram_usage()
    
    print("\n" + "="*50)
    print(f"📥 REQUEST MODEL: {model_id} (Executed with: {current_active})")
    print(f"⏱️ LATENCY: {latency}s | VRAM: {vram['allocated_mb']}MB")
    print("="*50 + "\n", flush=True)

    return {
        "success": True,
        "data": parsed,
        "raw": out_text,
        "latency_s": latency,
        "model_version": current_active,
        "vram": vram
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
