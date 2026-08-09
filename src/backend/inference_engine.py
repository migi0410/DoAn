import os
import warnings
from typing import List, Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

from baselines.baseline_rule_based import extract_kie_rules
from utils.preprocessing import ImagePreprocessor, TextPreprocessor

# PaddleOCR is imported lazily inside _initialize() to prevent module crash

warnings.filterwarnings("ignore")

class RuleModel:
    def predict(self, words, bboxes, img_path):
        return extract_kie_rules(words)

class PhoBertModel:
    def __init__(self, model_dir):
        import torch
        from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizerFast.from_pretrained("vinai/phobert-base-v2", add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        from utils.preprocessing import TextPreprocessor
        
        box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
        if preprocess_text:
            sorted_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in sorted_dicts]
            
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
        
        word_ids = encoding.word_ids()
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}
        
        \
        vocab_size = self.model.config.vocab_size
        encoding_gpu["input_ids"][encoding_gpu["input_ids"] >= vocab_size] = self.tokenizer.unk_token_id
        
        with torch.no_grad():
            outputs = self.model(**encoding_gpu)
            
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        _LABEL_MAP = {"ITEM_QUANTITY": "ITEM_QTY", "ITEM_TOTAL": "ITEM_AMOUNT"}
        def parse_labels_from_predictions(words, labels):
            parsed = {
                "SELLER": [], "ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": [],
                "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
            }
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = _LABEL_MAP.get(label[2:], label[2:])
                    if bio_tag == "B":
                        if current_entity["label"]:
                            parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
                        current_entity = {"label": entity_type, "words": [word]}
                    elif bio_tag == "I" and current_entity["label"] == entity_type:
                        current_entity["words"].append(word)
                else:
                    if current_entity["label"]:
                        parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
                        current_entity = {"label": None, "words": []}
            if current_entity["label"]:
                parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
            
            for k in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                parsed[k] = " ".join(parsed[k]).strip()
                
            items = []
            max_len = max(len(parsed["ITEM_NAME"]), len(parsed["ITEM_QTY"]), len(parsed["ITEM_PRICE"]), len(parsed["ITEM_AMOUNT"]))
            for i in range(max_len):
                items.append({
                    "ITEM_NAME": parsed["ITEM_NAME"][i] if i < len(parsed["ITEM_NAME"]) else "",
                    "ITEM_QTY": parsed["ITEM_QTY"][i] if i < len(parsed["ITEM_QTY"]) else "",
                    "ITEM_PRICE": parsed["ITEM_PRICE"][i] if i < len(parsed["ITEM_PRICE"]) else "",
                    "ITEM_AMOUNT": parsed["ITEM_AMOUNT"][i] if i < len(parsed["ITEM_AMOUNT"]) else ""
                })
            
            del parsed["ITEM_NAME"]
            del parsed["ITEM_QTY"]
            del parsed["ITEM_PRICE"]
            del parsed["ITEM_AMOUNT"]
            parsed["ITEMS"] = items
            return parsed

        return parse_labels_from_predictions(words, word_predicted_labels)

class LayoutLMModel:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoProcessor, AutoModelForTokenClassification
        # Use GPU, but we will disable cudnn for the forward pass to avoid the 'GET was unable to find an engine' error
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_dir)
        except Exception:
            self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        import cv2
        from utils.preprocessing import TextPreprocessor
        from PIL import Image
        
        box_dicts = [{"text": w, "box": b} for w, b in zip(words, bboxes)]
        if preprocess_text:
            box_dicts = TextPreprocessor.sort_reading_order(box_dicts)
            words = [item["text"] for item in box_dicts]
            bboxes = [item["box"] for item in box_dicts]
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        normalized_boxes = []
        for box in bboxes:
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            x_max = max([p[0] for p in box])
            y_max = max([p[1] for p in box])
            normalized_boxes.append([
                int(1000 * (x_min / w)),
                int(1000 * (y_min / h)),
                int(1000 * (x_max / w)),
                int(1000 * (y_max / h))
            ])
            
        image_pil = Image.open(img_path).convert("RGB")
            
        encoding = self.processor(
            image_pil, words, boxes=normalized_boxes, return_tensors="pt", truncation=True, max_length=512
        )
        
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            # Disable cudnn explicitly for this forward pass to avoid CUDA engine crash
            with torch.backends.cudnn.flags(enabled=False):
                outputs = self.model(**encoding_gpu)
            
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids()
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        _LABEL_MAP = {"ITEM_QUANTITY": "ITEM_QTY", "ITEM_TOTAL": "ITEM_AMOUNT"}
        def parse_labels_from_predictions(words, labels):
            parsed = {
                "SELLER": [], "ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": [],
                "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
            }
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = _LABEL_MAP.get(label[2:], label[2:])
                    if bio_tag == "B":
                        if current_entity["label"]:
                            parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
                        current_entity = {"label": entity_type, "words": [word]}
                    elif bio_tag == "I" and current_entity["label"] == entity_type:
                        current_entity["words"].append(word)
                else:
                    if current_entity["label"]:
                        parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
                        current_entity = {"label": None, "words": []}
            if current_entity["label"]:
                parsed[current_entity["label"]].append(" ".join(current_entity["words"]))
            
            for k in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                parsed[k] = " ".join(parsed[k]).strip()
                
            items = []
            max_len = max(len(parsed["ITEM_NAME"]), len(parsed["ITEM_QTY"]), len(parsed["ITEM_PRICE"]), len(parsed["ITEM_AMOUNT"]))
            for i in range(max_len):
                items.append({
                    "ITEM_NAME": parsed["ITEM_NAME"][i] if i < len(parsed["ITEM_NAME"]) else "",
                    "ITEM_QTY": parsed["ITEM_QTY"][i] if i < len(parsed["ITEM_QTY"]) else "",
                    "ITEM_PRICE": parsed["ITEM_PRICE"][i] if i < len(parsed["ITEM_PRICE"]) else "",
                    "ITEM_AMOUNT": parsed["ITEM_AMOUNT"][i] if i < len(parsed["ITEM_AMOUNT"]) else ""
                })
            
            del parsed["ITEM_NAME"]
            del parsed["ITEM_QTY"]
            del parsed["ITEM_PRICE"]
            del parsed["ITEM_AMOUNT"]
            parsed["ITEMS"] = items
            return parsed

        return parse_labels_from_predictions(words, word_predicted_labels)

class Qwen2VLModelWrapper:
    def __init__(self, model_dir):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            import json
            with open(os.path.join(model_dir, "adapter_config.json")) as f:
                adapter_config = json.load(f)
                base_model_id = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
                if base_model_id.startswith("/root/") or base_model_id.startswith("/workspace/"):
                    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
        except:
            base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
            
        print(f"Loading Qwen2-VL Base: {base_model_id}")
        import torch
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id, device_map="auto", torch_dtype=torch.float16
        )
        
        if os.path.exists(model_dir):
            print(f"Loading Qwen2-VL LoRA: {model_dir}")
            
            \
            adapter_path = os.path.join(model_dir, "adapter_config.json")
            if os.path.exists(adapter_path):
                import json
                try:
                    with open(adapter_path, 'r') as f:
                        cfg = json.load(f)
                    changed = False
                    whitelist = {
                        "peft_type", "auto_mapping", "base_model_name_or_path", "revision",
                        "task_type", "inference_mode", "r", "target_modules", "lora_alpha",
                        "lora_dropout", "fan_in_fan_out", "bias", "modules_to_save",
                        "init_lora_weights", "layers_to_transform", "layers_pattern",
                        "rank_pattern", "alpha_pattern", "megatron_config", "megatron_core",
                        "loftq_config", "use_rslora", "use_dora", "layer_replication", "target_parameters"
                    }
                    
                    keys_to_delete = [k for k in cfg.keys() if k not in whitelist]
                    for key in keys_to_delete:
                        del cfg[key]
                        changed = True
                    
                    if "target_modules" in cfg and isinstance(cfg["target_modules"], str):
                        \
\
                        cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                        changed = True
                        print(f"Converted target_modules from regex to list")
                        
                    if cfg.get("base_model_name_or_path", "").startswith("/root/") or cfg.get("base_model_name_or_path", "").startswith("/workspace/"):
                        cfg["base_model_name_or_path"] = "Qwen/Qwen2-VL-2B-Instruct"
                        changed = True
                        
                    if changed:
                        with open(adapter_path, 'w') as f:
                            json.dump(cfg, f, indent=2)
                            
                    safetensors_path = os.path.join(model_dir, "adapter_model.safetensors")
                    if os.path.exists(safetensors_path):
                        from safetensors.torch import load_file, save_file
                        import re as _re
                        try:
                            sd = load_file(safetensors_path)
                            sample_key = next(iter(sd.keys()))
                            needs_patch = "language_model" in sample_key or not ".default." in sample_key
                            
                            if needs_patch:
                                print(f"Patching safetensors keys...")
                                print(f"  Before: {sample_key}")
                                new_sd = {}
                                for k, v in sd.items():
                                    new_k = k
                                    \
                                    new_k = new_k.replace(".language_model.", ".")
                                    \
                                    new_k = _re.sub(r'\.(lora_[AB])\.weight$', r'.\1.default.weight', new_k)
                                    new_sd[new_k] = v
                                
                                sample_new_key = next(iter(new_sd.keys()))
                                print(f"  After:  {sample_new_key}")
                                
                                import shutil
                                shutil.copy(safetensors_path, safetensors_path + ".bak_orig")
                                save_file(new_sd, safetensors_path)
                                print("Safetensors keys patched successfully!")
                            else:
                                print(f"Safetensors keys already correct: {sample_key}")
                        except Exception as e:
                            print(f"Failed to patch safetensors: {e}")
                            
                except:
                    pass
                    
            from peft import PeftModel
            try:
                self.model = PeftModel.from_pretrained(self.model, model_dir)
                print(f"Qwen2-VL LoRA loaded successfully from {model_dir}")
            except Exception as lora_err:
                print(f"WARNING: Qwen2-VL LoRA FAILED to load: {lora_err}\nRunning base model only!")
            
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.model.eval()

    def generate_response(self, img_path, prompt):
        import torch
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.05)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        return output_text[0]

    def chat(self, img_path, question):
        return self.generate_response(img_path, question)

    def predict(self, img_path):
        prompt = "Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON. TUYỆT ĐỐI CHỈ lấy thông tin có trong ảnh, KHÔNG ĐƯỢC tự bịa thêm dữ liệu, KHÔNG giải thích."
        import json, re
        response = self.generate_response(img_path, prompt)
        print(f"================ VLM RAW RESPONSE ================\n{response}\n================================================")
        try:
            with open("debug_vlm.txt", "w", encoding="utf-8") as f:
                f.write(response)
        except: pass
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].strip()
            
            start_idx = response.find("{")
            if start_idx != -1:
                end_idx = response.rfind("}")
                if end_idx > start_idx:
                    response = response[start_idx:end_idx+1]
            
            parsed = json.loads(response)
            parsed = {k.upper(): v for k, v in parsed.items()}
            
            for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                if key not in parsed:
                    parsed[key] = ""
            
            if "ITEMS" not in parsed:
                items = []
                item_names = parsed.get("ITEM_NAME", [])
                item_qtys = parsed.get("ITEM_QTY", [])
                item_prices = parsed.get("ITEM_PRICE", [])
                item_amounts = parsed.get("ITEM_AMOUNT", [])
                
                if not isinstance(item_names, list): item_names = [item_names]
                if not isinstance(item_qtys, list): item_qtys = [item_qtys]
                if not isinstance(item_prices, list): item_prices = [item_prices]
                if not isinstance(item_amounts, list): item_amounts = [item_amounts]
                
                num_items = max(len(item_names), len(item_qtys), len(item_prices), len(item_amounts))
                for i in range(num_items):
                    items.append({
                        "ITEM_NAME": item_names[i] if i < len(item_names) else "",
                        "ITEM_QTY": item_qtys[i] if i < len(item_qtys) else "",
                        "ITEM_PRICE": item_prices[i] if i < len(item_prices) else "",
                        "ITEM_AMOUNT": item_amounts[i] if i < len(item_amounts) else "",
                    })
                parsed["ITEMS"] = items
            return parsed
        except Exception as e:
            print(f"Failed to parse VLM response: {e}")
            return {}

class MiniCPMVModelWrapper:
    """Wrapper to communicate with the isolated MiniCPM-V server"""
    SERVER_URL = "http://127.0.0.1:8005"

    def __init__(self, model_dir):
        import subprocess, sys, time, requests as _req, os
        self._proc = None
        
        self.session = _req.Session()
        self.session.trust_env = False
        
        venv_python = "/workspace/minicpm_env/bin/python3"
        server_script = os.path.join(os.path.dirname(__file__), "minicpm_server.py")

        if not os.path.exists(venv_python):
            raise RuntimeError(
                "MiniCPM venv not found. Run setup_minicpm.sh first:\n"
                "  bash /workspace/DoAn/setup_minicpm.sh"
            )

        # Check if server already running
        try:
            r = self.session.get(f"{self.SERVER_URL}/health", timeout=2)
            if r.ok:
                print("[MiniCPM] Server already running.")
                return
        except Exception:
            pass

        # Start server as subprocess and pipe logs to terminal
        env = {**os.environ, "MINICPM_MODEL_DIR": model_dir}
        self._proc = subprocess.Popen(
            [venv_python, server_script],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("[MiniCPM] Starting subprocess server...")
        # Wait up to 600s for server to be ready
        for _ in range(300):
            time.sleep(2)
            try:
                r = self.session.get(f"{self.SERVER_URL}/health", timeout=2)
                if r.ok:
                    print("[MiniCPM] Subprocess server ready!")
                    return
            except Exception:
                pass
        raise RuntimeError("MiniCPM server did not start in time (waited 600s).")
        
    def shutdown(self):
        """Kills the subprocess to free VRAM"""
        if self._proc is not None:
            print("Terminating MiniCPM-V server...")
            try:
                import psutil
                parent = psutil.Process(self._proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass
            self._proc.terminate()
            self._proc = None
        else:
            # Maybe it was running from a previous instance, try to kill port 8005
            import subprocess
            try:
                subprocess.run(["fuser", "-k", "8005/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except:
                pass

    def predict(self, img_path):
        import requests, os
        session = requests.Session()
        session.trust_env = False
        with open(img_path, "rb") as f:
            fname = os.path.basename(img_path)
            r = session.post(
                f"{self.SERVER_URL}/predict",
                files={"file": (fname, f, "image/jpeg")},
                timeout=300,
            )
        if not r.ok:
            raise RuntimeError(f"MiniCPM proxy error {r.status_code}: {r.text}")
        return r.json()

    def chat(self, img_path, question):
        import requests, os
        session = requests.Session()
        session.trust_env = False
        with open(img_path, "rb") as f:
            fname = os.path.basename(img_path)
            r = session.post(
                f"{self.SERVER_URL}/predict",
                files={"file": (fname, f, "image/jpeg")},
                timeout=300,
            )
        if not r.ok:
            raise RuntimeError(f"MiniCPM proxy error {r.status_code}: {r.text}")
        data = r.json()
        return str(data)

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        import torch
        print("Initializing Model Registry...")
        _gpu = torch.cuda.is_available()
        print(f"Initializing OCR (gpu={_gpu})...")
        self.ocr_paddle = None
        self._ocr_backend = None
        # Try EasyOCR first (no C++ MKL-DNN issues)
        try:
            import easyocr
            self.ocr_paddle = easyocr.Reader(['vi', 'en'], gpu=_gpu)
            self._ocr_backend = 'easyocr'
            print("EasyOCR initialized.")
        except Exception as e:
            print(f"EasyOCR unavailable ({e}), trying PaddleOCR...")
            try:
                from paddleocr import PaddleOCR
                self.ocr_paddle = PaddleOCR(use_angle_cls=False, lang="vi",
                                            device="gpu" if _gpu else "cpu")
                self._ocr_backend = 'paddleocr'
                print("PaddleOCR initialized.")
            except Exception as e2:
                print(f"PaddleOCR also unavailable ({e2}). OCR disabled.")
        self.rule_model = None
        self.phobert_model = None
        self.layoutlm_model = None
        self.qwen_model = None
        self.minicpm_model = None
        
        self.craft = None
        self.vietocr_detector = None
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        if os.path.exists("/workspace"): 
            self.models_dir = "/workspace"
        else:
            self.models_dir = os.path.join(base_dir, "trained_models")
            
        print(f"Models Directory configured to: {self.models_dir}")
        print("Eager loading default models...")
        for model_name in ["rule_based", "phobert", "layoutlmv3", "qwen2_vl"]:
            try:
                self.get_model(model_name)
                print(f"  ✓ {model_name} loaded")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ {model_name} failed to load: {e}")
        print("Model Registry ready!")

    def get_model(self, model_name):
        import gc
        import torch
        if model_name in ["qwen2_vl", "minicpm_v"]:
            # If requesting Qwen2-VL, unload MiniCPM-V if it exists
            if model_name == "qwen2_vl" and self.minicpm_model is not None:
                print("Lazy Unloading MiniCPM-V to save VRAM...")
                try:
                    self.minicpm_model.shutdown()
                except: pass
                del self.minicpm_model
                self.minicpm_model = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            # If requesting MiniCPM-V, unload Qwen2-VL if it exists
            if model_name == "minicpm_v" and self.qwen_model is not None:
                print("Lazy Unloading Qwen2-VL to save VRAM...")
                del self.qwen_model
                self.qwen_model = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
        if model_name == "rule_based":
            if self.rule_model is None:
                print("Lazy Loading Rule-based Model...")
                self.rule_model = RuleModel()
            return self.rule_model
        elif model_name == "phobert":
            if self.phobert_model is None:
                print("Lazy Loading PhoBERT...")
                path0 = os.path.join(self.models_dir, "phobert_avir_official_best")
                path1 = os.path.join(self.models_dir, "DoAn", "3_models_new", "phobert_best_model")
                path2 = os.path.join(self.models_dir, "3_models_new", "phobert_best_model")
                path3 = os.path.join(self.models_dir, "phobert_best_model")
                path = path0 if os.path.exists(path0) else path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else path3 if os.path.exists(path3) else os.path.join(self.models_dir, "phobert-base-kie")
                self.phobert_model = PhoBertModel(path)
            return self.phobert_model
        elif model_name == "layoutlmv3":
            if self.layoutlm_model is None:
                print("Lazy Loading LayoutLMv3...")
                path0 = os.path.join(self.models_dir, "layoutlmv3_avir_official_best")
                path1 = os.path.join(self.models_dir, "DoAn", "3_models_new", "layoutlmv3_best_model")
                path2 = os.path.join(self.models_dir, "3_models_new", "layoutlmv3_best_model")
                path3 = os.path.join(self.models_dir, "layoutlmv3_best_model")
                path = path0 if os.path.exists(path0) else path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else path3 if os.path.exists(path3) else os.path.join(self.models_dir, "layoutlm-avir-kie-best-10k")
                self.layoutlm_model = LayoutLMModel(path)
            return self.layoutlm_model
        elif model_name == "qwen2_vl":
            if self.qwen_model is None:
                print("Lazy Loading Qwen2-VL...")
                \
                path_official  = os.path.join(self.models_dir, "qwen2_vl_lora_official", "v7-20260808-192851", "checkpoint-582")
                path_official2 = os.path.join(self.models_dir, "qwen2_vl_lora_official")
                path_legacy0   = os.path.join(self.models_dir, "qwen2_vl_lora_swift", "v8-20260807-040045", "checkpoint-729")
                path = (path_official  if os.path.exists(path_official)
                        else path_official2 if os.path.exists(path_official2)
                        else path_legacy0   if os.path.exists(path_legacy0)
                        else path_official)
                print(f"Qwen2-VL path: {path}")
                self.qwen_model = Qwen2VLModelWrapper(path)
            return self.qwen_model
        elif model_name == "minicpm_v":
            if self.minicpm_model is None:
                print("Lazy Loading MiniCPM-V (via subprocess server)...")
                path_official = os.path.join(self.models_dir, "minicpm_v_lora_official")
                path_legacy0  = os.path.join(self.models_dir, "checkpoint-728")
                path_legacy1  = os.path.join(self.models_dir, "minicpm_lora_swift")
                path = (path_official if os.path.exists(path_official)
                        else path_legacy0 if os.path.exists(path_legacy0)
                        else path_legacy1 if os.path.exists(path_legacy1)
                        else path_official)
                print(f"MiniCPM-V path: {path}")
                self.minicpm_model = MiniCPMVModelWrapper(path)
            return self.minicpm_model
        return None

    def run_paddle_ocr(self, img_path):
        print("Running OCR...")
        words, bboxes = [], []
        if self.ocr_paddle is None:
            print("OCR unavailable - returning empty.")
            return words, bboxes
        if getattr(self, '_ocr_backend', 'paddleocr') == 'easyocr':
            result = self.ocr_paddle.readtext(img_path)
            for (box, text, conf) in result:
                try:
                    words.append(text)
                    # box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                    bboxes.append([[int(p[0]), int(p[1])] for p in box])
                except Exception:
                    pass
            return words, bboxes
        # PaddleOCR fallback
        try:
            result = self.ocr_paddle.ocr(img_path)
        except TypeError:
            result = self.ocr_paddle.ocr(img_path, cls=False)
        if not result:
            return words, bboxes
        page = result[0] if isinstance(result, list) else result
        if isinstance(page, dict):
            for text, box in zip(page.get("rec_texts", []), page.get("rec_boxes", [])):
                words.append(text)
                bboxes.append(box.tolist() if hasattr(box, "tolist") else box)
        elif isinstance(page, list):
            for item in page:
                try:
                    if isinstance(item, dict):
                        words.append(item.get("rec_text", ""))
                        box = item.get("rec_box", [])
                        bboxes.append(box.tolist() if hasattr(box, "tolist") else box)
                    else:
                        box, (text, _conf) = item[0], item[1]
                        if isinstance(box, (list, tuple)):
                            bboxes.append(box)
                            words.append(text)
                except Exception:
                    pass
        return words, bboxes

    def _init_craft_vietocr(self):
        if self.craft is None:
            print("Lazy Loading CRAFT and VietOCR...")
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                from craft_text_detector import Craft
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
            except ImportError:
                print("Missing craft-text-detector or vietocr.")
                return False
                
            self.craft = Craft(output_dir=None, crop_type="box", cuda=(device == 'cuda'))
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = False
            config['device'] = device
            self.vietocr_detector = Predictor(config)
        return True

    def run_craft_vietocr(self, img_path):
        import cv2
        import numpy as np
        from PIL import Image
        
        if not self._init_craft_vietocr():
            print("Warning: CRAFT + VietOCR not installed. Falling back to PaddleOCR.")
            return self.run_paddle_ocr(img_path)
            
        print("Running CRAFT + VietOCR...")
        image_cv = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        prediction_result = self.craft.detect_text(img_path)
        boxes = prediction_result["boxes"]
        
        words, bboxes = [], []
        
        def crop_poly(img, pts):
            rect = cv2.boundingRect(pts.astype(np.int32))
            x, y, w, h = rect
            pad = 2
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + pad*2)
            h = min(img.shape[0] - y, h + pad*2)
            return img[y:y+h, x:x+w]
            
        for box in boxes:
            cropped_cv = crop_poly(image_rgb, box)
            if cropped_cv.shape[0] == 0 or cropped_cv.shape[1] == 0:
                continue
            cropped_pil = Image.fromarray(cropped_cv)
            try:
                text = self.vietocr_detector.predict(cropped_pil)
            except:
                text = ""
                
            if text.strip():
                words.append(text)
                bboxes.append(box.tolist())
                
        return words, bboxes

    def predict(self, baseline, img_path, preprocess=False):
        words, bboxes = [], []
        result = {}
        
        if baseline in ["qwen2_vl", "minicpm_v"]:
            pass
        elif baseline in ["phobert_paddle", "layoutlmv3", "rule_based"]:
            words, bboxes = self.run_paddle_ocr(img_path)
        elif baseline in ["phobert", "layoutlmv3_craft"]:
            # Try CRAFT first, falls back to PaddleOCR/EasyOCR
            words, bboxes = self.run_craft_vietocr(img_path)
        else:
            words, bboxes = self.run_paddle_ocr(img_path)

        if baseline == "rule_based":
            model = self.get_model("rule_based")
            if model:
                result = model.predict(words, bboxes, img_path)
        elif baseline in ["phobert", "phobert_paddle"]:
            model = self.get_model("phobert")
            if model:
                result = model.predict(words, bboxes, img_path, preprocess_text=preprocess)
        elif baseline in ["layoutlmv3", "layoutlmv3_craft"]:
            model = self.get_model("layoutlmv3")
            if model:
                result = model.predict(words, bboxes, img_path, preprocess_text=preprocess)
        elif baseline == "qwen2_vl":
            model = self.get_model("qwen2_vl")
            if model:
                result = model.predict(img_path)
        elif baseline == "minicpm_v":
            model = self.get_model("minicpm_v")
            if model:
                result = model.predict(img_path)
                
        for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
            if key not in result:
                result[key] = ""
            elif isinstance(result[key], dict):
                # If the model hallucinated an object, try to extract its name/value, or just stringify it
                val = result[key]
                result[key] = str(val.get("name", val.get("value", val)))
            elif isinstance(result[key], list):
                result[key] = str(result[key][0]) if result[key] else ""
            else:
                result[key] = str(result[key])
        
        if "ITEMS" not in result or not isinstance(result["ITEMS"], list):
            result["ITEMS"] = []
                
        return result, words, bboxes

    def chat(self, model_name, img_path, question):
        model = self.get_model(model_name)
        if model:
            if hasattr(model, "chat"):
                return model.chat(img_path, question)
            else:
                return f"Model {model_name} does not support Chat/VQA."
        return "Model not supported or not loaded."
