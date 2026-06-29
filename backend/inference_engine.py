import os
import sys
import torch
import cv2
import warnings

# Thêm đường dẫn tới pipeline để import các lớp model đã có sẵn nếu cần.
# Hoặc ta sẽ sao chép logic import ở đây.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines_and_training.baselines.baseline_rule_based import extract_kie_rules
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


class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        print("Initializing Model Registry...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4")
        
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_models"))
        
        self.rule_model = RuleModel()
        
        try:
            print("Loading PhoBERT...")
            self.phobert_model = PhoBertModel(os.path.join(models_dir, "phobert-avir-kie-best-10k"))
        except Exception as e:
            print(f"Error loading PhoBERT: {e}")
            self.phobert_model = None
            
        try:
            print("Loading LayoutLM...")
            self.layoutlm_model = LayoutLMModel(os.path.join(models_dir, "layoutlm-avir-kie-best-10k"))
        except Exception as e:
            print(f"Error loading LayoutLM: {e}")
            self.layoutlm_model = None

        print("Model Registry Initialized Successfully!")

    def run_paddle_ocr(self, img_path):
        result = self.ocr.ocr(img_path, cls=False)
        words, bboxes = [], []
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                bboxes.append(box)
                words.append(text)
        return words, bboxes

    def predict(self, baseline, img_path, preprocess=False):
        # 1. OCR
        words, bboxes = self.run_paddle_ocr(img_path)
        
        # 2. Select Model & Run Inference
        result = {}
        if baseline == "rule_paddle":
            result = self.rule_model.predict(words, bboxes, img_path)
        elif baseline == "phobert":
            if self.phobert_model:
                result = self.phobert_model.predict(words, bboxes, img_path, preprocess_text=preprocess)
        elif baseline == "layoutlmv1":
            if self.layoutlm_model:
                result = self.layoutlm_model.predict(words, bboxes, img_path, preprocess_text=preprocess)
                
        # Fill missing keys if any
        for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
            if key not in result:
                result[key] = ""
                
        return result, words, bboxes
