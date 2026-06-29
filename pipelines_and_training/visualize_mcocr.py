import os
import json
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = LayoutLMTokenizerFast.from_pretrained("../trained_models/layoutlm-avir-kie-best-10k")
model = LayoutLMForTokenClassification.from_pretrained("../trained_models/layoutlm-avir-kie-best-10k")
model.to(device)
model.eval()

LABELS = ["O", "B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST", "B-ITEM_NAME", "I-ITEM_NAME", "B-ITEM_QTY", "I-ITEM_QTY", "B-ITEM_PRICE", "I-ITEM_PRICE", "B-ITEM_AMOUNT", "I-ITEM_AMOUNT", "B-OTHER", "I-OTHER"]
id2label = {idx: lbl for idx, lbl in enumerate(LABELS)}
LABEL_COLORS = {"SELLER": "red", "ADDRESS": "blue", "TIMESTAMP": "green", "TOTAL_COST": "orange", "OTHER": "purple", "ITEM_NAME": "yellow", "ITEM_QTY": "cyan", "ITEM_PRICE": "magenta", "ITEM_AMOUNT": "brown"}

def normalize_bbox(bbox, width, height):
    return [max(0, min(1000, int(1000 * (bbox[0] / width)))), max(0, min(1000, int(1000 * (bbox[1] / height)))), max(0, min(1000, int(1000 * (bbox[2] / width)))), max(0, min(1000, int(1000 * (bbox[3] / height))))]

def unnormalize_bbox(bbox, width, height):
    return [int(width * (bbox[0] / 1000)), int(height * (bbox[1] / 1000)), int(width * (bbox[2] / 1000)), int(height * (bbox[3] / 1000))]

def split_box_into_words(text, box):
    words = text.split()
    if not words: return [], []
    x1, y1, x2, y2 = box
    total_len = max(1, sum(len(w) for w in words))
    word_boxes = []
    current_x = x1
    for w in words:
        w_width = (len(w) / total_len) * (x2 - x1)
        word_boxes.append([current_x, y1, current_x + w_width, y2])
        current_x += w_width + (x2 - x1)*0.02
    return words, word_boxes

img_path = r"C:\Users\Admin\OneDrive\DoAn\raw_data\mcocr\val_images\val_images\mcocr_val_145114gpkup.jpg"
cache = json.load(open("ocr_cache.json", encoding="utf-8"))
ocr_results = next((v for k, v in cache.items() if "mcocr_val_145114gpkup.jpg" in k), None)

img = Image.open(img_path).convert("RGB")
width, height = img.size
words, bboxes = [], []
for item in ocr_results:
    box = item[0]
    w_list, box_list = split_box_into_words(item[1][0], [min(p[0] for p in box), min(p[1] for p in box), max(p[0] for p in box), max(p[1] for p in box)])
    for w, b in zip(w_list, box_list):
        words.append(w); bboxes.append(normalize_bbox(b, width, height))

encoding = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
word_ids_list = encoding.word_ids(batch_index=0)
bbox_tensors = [[0, 0, 0, 0] if i is None else bboxes[i] for i in word_ids_list]
encoding["bbox"] = torch.tensor([bbox_tensors])
del encoding["offset_mapping"]
encoding = {k: v.to(device) for k, v in encoding.items()}

with torch.no_grad(): outputs = model(**encoding)
predictions = outputs.logits.argmax(-1).squeeze().tolist()
if isinstance(predictions, int): predictions = [predictions]
token_boxes = encoding["bbox"].squeeze().tolist()

draw = ImageDraw.Draw(img)
try: font = ImageFont.truetype("arial.ttf", 20)
except: font = ImageFont.load_default()

prev_word_idx = None
for idx, (pred, word_idx) in enumerate(zip(predictions, word_ids_list)):
    if word_idx is None or word_idx == prev_word_idx: continue
    label = id2label[pred]
    if label not in ["O", "B-OTHER", "I-OTHER"]:
        color = LABEL_COLORS.get(label[2:] if "-" in label else label, "purple")
        actual_box = unnormalize_bbox(token_boxes[idx], width, height)
        draw.rectangle(actual_box, outline=color, width=3)
        draw.text((actual_box[0], actual_box[1]-20), label, fill=color, font=font)
    prev_word_idx = word_idx

img.save("mcocr_inference_sample.png")
print("Done")
