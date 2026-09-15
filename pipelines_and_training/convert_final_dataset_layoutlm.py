import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict
from tqdm import tqdm

LABELS = [
    "O",
    "B-SELLER", "I-SELLER",
    "B-ADDRESS", "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP",
    "B-TOTAL_COST", "I-TOTAL_COST",
    "B-ITEM_NAME", "I-ITEM_NAME",
    "B-ITEM_QTY", "I-ITEM_QTY",
    "B-ITEM_PRICE", "I-ITEM_PRICE",
    "B-ITEM_AMOUNT", "I-ITEM_AMOUNT",
    "B-OTHER", "I-OTHER"
]
label2id = {label: i for i, label in enumerate(LABELS)}

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height))
    ]

def split_box_into_words(text, box):
    words = text.split()
    if not words:
        return [], []
        
    num_words = len(words)
    x1, y1, x2, y2 = box
    
    total_width = x2 - x1
    word_width = total_width / num_words
    
    word_boxes = []
    current_x = x1
    for i in range(num_words):
        next_x = current_x + word_width
        word_boxes.append([int(current_x), int(y1), int(next_x), int(y2)])
        current_x = next_x
        
    return words, word_boxes

def process_dataset(json_path, image_base_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    splits = {"train": [], "val": [], "test": []}
    
    for item in tqdm(data, desc="Processing dataset"):
        file_name = item["file_name"]
        
        # Determine split
        if "_train_" in file_name:
            split = "train"
        elif "_val_" in file_name:
            split = "val"
        elif "_test_" in file_name:
            split = "test"
        else:
            continue
            
        image_path = os.path.join(image_base_dir, file_name)
        if not os.path.exists(image_path):
            continue
            
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue
            
        tokens = []
        bboxes = []
        ner_tags = []
        
        for ann in item.get("annotations", []):
            label = ann["label"]
            text = ann["text"]
            box = ann["box"]
            
            if not text.strip():
                continue
                
            words, word_boxes = split_box_into_words(text, box)
            if not words:
                continue
                
            for i, (word, w_box) in enumerate(zip(words, word_boxes)):
                # Normalize bbox
                norm_box = normalize_bbox(w_box, width, height)
                
                # Clip to [0, 1000]
                norm_box = [max(0, min(1000, coord)) for coord in norm_box]
                
                # Generate label
                if label == "OTHER":
                    tag = f"B-OTHER" if i == 0 else f"I-OTHER"
                elif label in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"]:
                    tag = f"B-{label}" if i == 0 else f"I-{label}"
                else:
                    tag = "O"
                    
                tokens.append(word)
                bboxes.append(norm_box)
                ner_tags.append(label2id.get(tag, 0))
                
        if len(tokens) > 0:
            splits[split].append({
                "id": str(len(splits[split])),
                "tokens": tokens,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image_path": image_path
            })
            
    # Convert to HuggingFace Dataset
    hf_splits = {}
    for k, v in splits.items():
        if len(v) > 0:
            hf_splits[k] = Dataset.from_list(v)
            
    hf_dataset = DatasetDict(hf_splits)
    return hf_dataset

if __name__ == "__main__":
    base_dir = r"c:\Users\Admin\OneDrive\DoAn"
    json_path = os.path.join(base_dir, "FINAL_BBOX_DATASET_V3.json")
    image_base_dir = os.path.join(base_dir, "FINAL_RUNPOD_DATASET")
    output_dir = os.path.join(base_dir, "FINAL_LAYOUTLM_DATASET")
    
    print("Bắt đầu xử lý dữ liệu...")
    hf_dataset = process_dataset(json_path, image_base_dir)
    print(f"Hoàn thành! Kích thước: Train={len(hf_dataset.get('train', []))}, Val={len(hf_dataset.get('val', []))}, Test={len(hf_dataset.get('test', []))}")
    
    os.makedirs(output_dir, exist_ok=True)
    hf_dataset.save_to_disk(output_dir)
    print(f"Đã lưu dataset HuggingFace tại {output_dir}")
