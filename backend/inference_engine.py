import os
import torch
import cv2
import warnings

# Thêm đường dẫn tới pipeline để import các lớp model đã có sẵn nếu cần.
# Hoặc ta sẽ sao chép logic import ở đây.

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
            parsed = {"SELLER": "", "ADDRESS": "", "TIMESTAMP": "", "TOTAL_COST": ""}
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = label[2:]
                    if bio_tag == "B":
                        if current_entity["label"]:
                            if current_entity["label"] in parsed:
                                parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
                        current_entity = {"label": entity_type, "words": [word]}
                    elif bio_tag == "I" and current_entity["label"] == entity_type:
                        current_entity["words"].append(word)
                else:
                    if current_entity["label"]:
                        if current_entity["label"] in parsed:
                            parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
                        current_entity = {"label": None, "words": []}
            if current_entity["label"] and current_entity["label"] in parsed:
                parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
            for k in parsed:
                parsed[k] = parsed[k].strip()
            return parsed

        return parse_labels_from_predictions(words, word_predicted_labels)

class LayoutLMModel:
    def __init__(self, model_dir):
        import torch
        from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
        self.model = LayoutLMForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, words, bboxes, img_path, preprocess_text=False):
        import torch
        import cv2
        from utils.preprocessing import TextPreprocessor
        
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
            
        encoding = self.tokenizer(
            words, boxes=normalized_boxes, return_tensors="pt", truncation=True, max_length=512
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
            parsed = {"SELLER": "", "ADDRESS": "", "TIMESTAMP": "", "TOTAL_COST": ""}
            current_entity = {"label": None, "words": []}
            for word, label in zip(words, labels):
                if label != "O":
                    bio_tag = label[0]
                    entity_type = label[2:]
                    if bio_tag == "B":
                        if current_entity["label"]:
                            if current_entity["label"] in parsed:
                                parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
                        current_entity = {"label": entity_type, "words": [word]}
                    elif bio_tag == "I" and current_entity["label"] == entity_type:
                        current_entity["words"].append(word)
                else:
                    if current_entity["label"]:
                        if current_entity["label"] in parsed:
                            parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
                        current_entity = {"label": None, "words": []}
            if current_entity["label"] and current_entity["label"] in parsed:
                parsed[current_entity["label"]] += " " + " ".join(current_entity["words"])
            for k in parsed:
                parsed[k] = parsed[k].strip()
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
        prompt = "Trích xuất thông tin hóa đơn dưới dạng JSON với các trường: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT, OTHER."
        import json
        response = self.generate_response(img_path, prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"OTHER": response}
            
    def chat(self, img_path, question):
        return self.generate_response(img_path, question)

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        print("Initializing Model Registry...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv3")
        self.rule_model = None
        self.phobert_model = None
        self.layoutlm_model = None
        self.qwen_model = None
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
        if not os.path.exists(self.models_dir):
            self.models_dir = os.path.dirname(os.path.dirname(__file__)) # For RunPod workspace)
        print("Model Registry Initialized (Lazy Loading mode)...")

    def get_model(self, baseline):
        if baseline == "rule_based":
            if self.rule_model is None:
                print("Lazy Loading Rule-based Model...")
                self.rule_model = RuleModel()
            return self.rule_model
        elif baseline == "phobert":
            if self.phobert_model is None:
                print("Lazy Loading PhoBERT...")
                self.phobert_model = PhoBertModel(os.path.join(self.models_dir, "phobert-base-kie"))
            return self.phobert_model
        elif baseline == "layoutlmv1":
            if self.layoutlm_model is None:
                print("Lazy Loading LayoutLM...")
                self.layoutlm_model = LayoutLMModel(os.path.join(self.models_dir, "layoutlm-avir-kie-best-10k"))
            return self.layoutlm_model
        elif baseline == "qwen2_vl":
            if self.qwen_model is None:
                print("Lazy Loading Qwen2-VL...")
                self.qwen_model = Qwen2VLModelWrapper(os.path.join(self.models_dir, "qwen2-vl-finetuned-lora"))
            return self.qwen_model
        return None

    def run_paddle_ocr(self, img_path):
        result = self.ocr.ocr(img_path, cls=False)
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

    def predict(self, baseline, img_path, preprocess=False):
        # 1. OCR
        words, bboxes = self.run_paddle_ocr(img_path)
        
        # 2. Select Model & Run Inference
        result = {}
        if baseline == "rule_based":
            model = self.get_model("rule_based")
            if model:
                result = model.predict(words, bboxes, img_path)
        elif baseline == "phobert":
            model = self.get_model("phobert")
            if model:
                result = model.predict(words, bboxes, img_path, preprocess_text=preprocess)
        elif baseline == "layoutlmv1":
            model = self.get_model("layoutlmv1")
            if model:
                result = model.predict(words, bboxes, img_path, preprocess_text=preprocess)
        elif baseline == "qwen2_vl":
            model = self.get_model("qwen2_vl")
            if model:
                result = model.predict(img_path)
                
        # Fill missing keys if any
        for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
            if key not in result:
                result[key] = ""
                
        return result, words, bboxes

    def chat(self, model_name, img_path, question):
        if model_name == "qwen2_vl":
            model = self.get_model("qwen2_vl")
            if model:
                return model.chat(img_path, question)
        return "Model not supported or not loaded."
