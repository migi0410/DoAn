import os
import torch
import cv2
import warnings

from baselines.baseline_rule_based import extract_kie_rules
from utils.preprocessing import ImagePreprocessor, TextPreprocessor
from paddleocr import PaddleOCR

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
        
        # Clamp out-of-vocabulary tokens
        vocab_size = self.model.config.vocab_size
        encoding_gpu["input_ids"][encoding_gpu["input_ids"] >= vocab_size] = self.tokenizer.unk_token_id
        
        with torch.no_grad():
            outputs = self.model(**encoding_gpu)
            
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        def parse_labels_from_predictions(words, labels):
            parsed = {
                "SELLER": [], "ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": [],
                "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
            }
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = label[2:]
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
            outputs = self.model(**encoding_gpu)
            
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids()
        
        word_predicted_labels = ["O"] * len(words)
        for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_predicted_labels[word_idx] == "O":
                word_predicted_labels[word_idx] = self.id2label[pred]
                
        def parse_labels_from_predictions(words, labels):
            parsed = {
                "SELLER": [], "ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": [],
                "ITEM_NAME": [], "ITEM_QTY": [], "ITEM_PRICE": [], "ITEM_AMOUNT": []
            }
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = label[2:]
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
                base_model_id = adapter_config.get("base_model_name_or_path", "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        except:
            base_model_id = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
            
        print(f"Loading Qwen2-VL Base: {base_model_id}")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id, 
                device_map="auto",
                quantization_config=quantization_config
            )
        except Exception as e:
            print(f"Failed to load with custom quantization config: {e}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id, device_map="auto"
            )
        
        if os.path.exists(model_dir):
            print(f"Loading Qwen2-VL LoRA: {model_dir}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_dir)
            
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
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
        prompt = "Trích xuất thông tin hóa đơn dưới dạng JSON với các trường: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, và mảng ITEMS gồm các object chứa (ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT)."
        import json
        response = self.generate_response(img_path, prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"OTHER": response}

class MiniCPMVModelWrapper:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            import json
            with open(os.path.join(model_dir, "adapter_config.json")) as f:
                adapter_config = json.load(f)
                base_model_id = adapter_config.get("base_model_name_or_path", "openbmb/MiniCPM-Llama3-V-2_5")
        except:
            base_model_id = "openbmb/MiniCPM-Llama3-V-2_5"
            
        print(f"Loading MiniCPM-V Base: {base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        
        if os.path.exists(model_dir):
            print(f"Loading MiniCPM-V LoRA: {model_dir}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_dir)
            
        self.model.eval()

    def generate_response(self, img_path, prompt):
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return res

    def chat(self, img_path, question):
        return self.generate_response(img_path, question)

    def predict(self, img_path):
        prompt = "Trích xuất thông tin hóa đơn dưới dạng JSON với các trường: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, và mảng ITEMS gồm các object chứa (ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT)."
        import json
        response = self.generate_response(img_path, prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"OTHER": response}

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        print("Initializing Model Registry...")
        self.ocr_paddle = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv3", use_gpu=False)
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
        print("Model Registry Initialized (Lazy Loading mode)...")

    def get_model(self, model_name):
        if model_name == "rule_based":
            if self.rule_model is None:
                print("Lazy Loading Rule-based Model...")
                self.rule_model = RuleModel()
            return self.rule_model
        elif model_name == "phobert":
            if self.phobert_model is None:
                print("Lazy Loading PhoBERT...")
                path1 = os.path.join(self.models_dir, "3_models_new", "phobert_best_model")
                path2 = os.path.join(self.models_dir, "phobert_best_model")
                path = path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else os.path.join(self.models_dir, "phobert-base-kie")
                self.phobert_model = PhoBertModel(path)
            return self.phobert_model
        elif model_name == "layoutlmv3":
            if self.layoutlm_model is None:
                print("Lazy Loading LayoutLMv3...")
                path1 = os.path.join(self.models_dir, "3_models_new", "layoutlmv3_best_model")
                path2 = os.path.join(self.models_dir, "layoutlmv3_best_model")
                path = path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else os.path.join(self.models_dir, "layoutlm-avir-kie-best-10k")
                self.layoutlm_model = LayoutLMModel(path)
            return self.layoutlm_model
        elif model_name == "qwen2_vl":
            if self.qwen_model is None:
                print("Lazy Loading Qwen2-VL...")
                path1 = os.path.join(self.models_dir, "3_models_new", "src_v2", "src_v2", "qwen2_vl_lora_swift")
                path2 = os.path.join(self.models_dir, "qwen2_vl_lora_v2")
                path = path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else os.path.join(self.models_dir, "qwen2-vl-finetuned-lora")
                self.qwen_model = Qwen2VLModelWrapper(path)
            return self.qwen_model
        elif model_name == "minicpm_v":
            if self.minicpm_model is None:
                print("Lazy Loading MiniCPM-V...")
                path1 = os.path.join(self.models_dir, "checkpoint-728")
                path2 = os.path.join(self.models_dir, "minicpm_lora_swift")
                path = path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else os.path.join(self.models_dir, "minicpm-v-finetuned")
                self.minicpm_model = MiniCPMVModelWrapper(path)
            return self.minicpm_model
        return None

    def run_paddle_ocr(self, img_path):
        print("Running PaddleOCR...")
        result = self.ocr_paddle.ocr(img_path, cls=False)
        words, bboxes = [], []
        if result and result[0]:
            for line in result[0]:
                try:
                    box = line[0]
                    text = line[1][0]
                    if isinstance(box, (list, tuple)) and not isinstance(box, str):
                        bboxes.append(box)
                        words.append(text)
                except:
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
        
        # 1. OCR Engine Selection
        if baseline in ["qwen2_vl", "minicpm_v"]:
            pass # No OCR needed
        elif baseline in ["phobert_paddle", "layoutlmv3", "rule_based"]:
            words, bboxes = self.run_paddle_ocr(img_path)
        elif baseline in ["phobert", "layoutlmv3_craft"]:
            words, bboxes = self.run_craft_vietocr(img_path)
        else:
            words, bboxes = self.run_paddle_ocr(img_path)

        # 2. Select Model & Run Inference
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
                
        # Fill missing keys if any
        for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
            if key not in result:
                result[key] = ""
        if "ITEMS" not in result:
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
