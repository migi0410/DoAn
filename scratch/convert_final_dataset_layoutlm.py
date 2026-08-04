import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
# OCR initialization is deferred to when it is actually needed

def calculate_iou(boxA, boxB):
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    box2_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(box1_area + box2_area - intersection_area)
    
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area

def normalize_bbox(box, width, height):
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height))))
    ]

def split_box_for_words(text, box):
    words = str(text).split()
    if not words: return [], []
    
    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    total_chars = sum(len(w) for w in words) + len(words) - 1
    char_width = box_width / max(1, total_chars)
    
    word_boxes = []
    current_x = x_min
    for w in words:
        w_width = max(1, int(len(w) * char_width))
        word_boxes.append([int(current_x), y_min, int(current_x + w_width), y_max])
        current_x += w_width + char_width
    return words, word_boxes

def convert_sample(img_path, ground_truths, mode="noise"):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Chạy OCR
    ocr_boxes = []
    ocr_words = []
    if mode != "clean_fast":
        global ocr
        if 'ocr' not in globals():
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv3")
        result = ocr.ocr(img_path)
        
        def extract_lines(res):
            extracted = []
            if isinstance(res, list):
                if len(res) == 2 and isinstance(res[0], list) and len(res[0]) == 4 and isinstance(res[1], tuple):
                    return [res]
                for item in res:
                    if item is not None:
                        extracted.extend(extract_lines(item))
            return extracted

        lines = extract_lines(result)
                
        for line in lines:
            box = line[0]
            text = line[1][0]
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            x_max = max([p[0] for p in box])
            y_max = max([p[1] for p in box])
            ocr_boxes.append([x_min, y_min, x_max, y_max])
            ocr_words.append(text)

    final_words = []
    final_boxes = []
    final_tags = []

    if mode == "both":
        # Do noise
        prev_label = "O"
        final_tags_noise = []
        for ocr_box, ocr_word in zip(ocr_boxes, ocr_words):
            best_iou = 0
            best_label = "O"
            for gt in ground_truths:
                iou = calculate_iou(ocr_box, gt["box"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_label = gt["label"]
            tag = "O"
            if best_label != "O":
                tag = f"B-{best_label}" if best_label != prev_label else f"I-{best_label}"
            final_tags_noise.append(tag)
            prev_label = best_label
        
        # Do clean
        final_words_clean = []
        final_boxes_clean = []
        final_tags_clean = []
        for gt in ground_truths:
            gt_words, gt_boxes = split_box_for_words(gt["text"], gt["box"])
            for i, (g_w, g_b) in enumerate(zip(gt_words, gt_boxes)):
                final_words_clean.append(g_w)
                final_boxes_clean.append(g_b)
                final_tags_clean.append(f"B-{gt['label']}" if i == 0 else f"I-{gt['label']}")
        
        for ocr_box, ocr_word in zip(ocr_boxes, ocr_words):
            is_overlap = False
            for gt in ground_truths:
                if calculate_iou(ocr_box, gt["box"]) > 0.1:
                    is_overlap = True
                    break
            if not is_overlap:
                final_words_clean.append(ocr_word)
                final_boxes_clean.append(ocr_box)
                final_tags_clean.append("O")

        return {
            "noise": {
                "image_path": img_path,
                "words": ocr_words,
                "bboxes": [normalize_bbox(b, w, h) for b in ocr_boxes],
                "ner_tags": final_tags_noise
            },
            "clean": {
                "image_path": img_path,
                "words": final_words_clean,
                "bboxes": [normalize_bbox(b, w, h) for b in final_boxes_clean],
                "ner_tags": final_tags_clean
            }
        }

    elif mode == "noise":
        # Mode 1: Sử dụng 100% chữ từ OCR (Tiêm nhiễu)
        prev_label = "O"
        for ocr_box, ocr_word in zip(ocr_boxes, ocr_words):
            best_iou = 0
            best_label = "O"
            
            for gt in ground_truths:
                iou = calculate_iou(ocr_box, gt["box"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_label = gt["label"]
                    
            tag = "O"
            if best_label != "O":
                tag = f"B-{best_label}" if best_label != prev_label else f"I-{best_label}"
            
            final_words.append(ocr_word)
            final_boxes.append(ocr_box)
            final_tags.append(tag)
            prev_label = best_label

    elif mode == "clean":
        # Mode 2: Sử dụng chữ chuẩn 100% từ Ground Truth
        # Bước 1: Thêm các Ground Truth chuẩn xác
        for gt in ground_truths:
            gt_words, gt_boxes = split_box_for_words(gt["text"], gt["box"])
            for i, (g_w, g_b) in enumerate(zip(gt_words, gt_boxes)):
                final_words.append(g_w)
                final_boxes.append(g_b)
                final_tags.append(f"B-{gt['label']}" if i == 0 else f"I-{gt['label']}")
        
        # Bước 2: Thêm các chữ rác từ OCR (nếu không đè lên Ground Truth)
        for ocr_box, ocr_word in zip(ocr_boxes, ocr_words):
            is_overlap = False
            for gt in ground_truths:
                if calculate_iou(ocr_box, gt["box"]) > 0.1:
                    is_overlap = True
                    break
            
            if not is_overlap:
                final_words.append(ocr_word)
                final_boxes.append(ocr_box)
                final_tags.append("O")

    elif mode == "clean_fast":
        for gt in ground_truths:
            gt_words, gt_boxes = split_box_for_words(gt["text"], gt["box"])
            for i, (g_w, g_b) in enumerate(zip(gt_words, gt_boxes)):
                final_words.append(g_w)
                final_boxes.append(g_b)
                final_tags.append(f"B-{gt['label']}" if i == 0 else f"I-{gt['label']}")
    normalized_boxes = [normalize_bbox(b, w, h) for b in final_boxes]
    
    return {
        "image_path": img_path,
        "words": final_words,
        "bboxes": normalized_boxes,
        "ner_tags": final_tags
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["noise", "clean", "clean_fast", "both"], required=True)
    parser.add_argument("--output", type=str, required=True); parser.add_argument("--input_dir", type=str, required=False, default="")
    args = parser.parse_args()
    
    base_dir = args.input_dir if args.input_dir else (r"/workspace" if os.path.exists("/workspace") else r"C:\Users\Admin\OneDrive\DoAn")
    gt_file = os.path.join(os.path.dirname(base_dir), "FINAL_BBOX_DATASET_V3.json") if os.path.exists(os.path.join(os.path.dirname(base_dir), "FINAL_BBOX_DATASET_V3.json")) else os.path.join(base_dir, "FINAL_BBOX_DATASET_V3.json")
    
    with open(gt_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Bắt đầu chuyển đổi {len(dataset)} ảnh ở chế độ: {args.mode.upper()}")
    
    import multiprocessing
    
    def process_wrapper(sample):
        img_path = os.path.join(base_dir, "FINAL_RUNPOD_DATASET", sample["file_name"])
        if not os.path.exists(img_path):
            return None
        try:
            return convert_sample(img_path, sample["annotations"], mode=args.mode)
        except Exception:
            return None

    results = []
    results_noise = []
    results_clean = []
    
    # Sử dụng tất cả số Core CPU hiện có trên máy (thường là 8-16 core trên RunPod)
    num_cores = min(multiprocessing.cpu_count(), 32)
    print(f"Khởi động Multiprocessing với {num_cores} CPU cores (đã giới hạn để tránh quá tải)...")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_wrapper, dataset), total=len(dataset)):
            if res:
                if args.mode == "both":
                    results_noise.append(res["noise"])
                    results_clean.append(res["clean"])
                else:
                    results.append(res)
                
    if args.mode == "both":
        with open("/workspace/FINAL_LAYOUTLM_NOISE.json", "w", encoding="utf-8") as f:
            json.dump(results_noise, f, ensure_ascii=False)
        with open("/workspace/FINAL_LAYOUTLM_CLEAN_FULL.json", "w", encoding="utf-8") as f:
            json.dump(results_clean, f, ensure_ascii=False)
        print("Đã lưu thành công cả 2 file NOISE và CLEAN_FULL!")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"Đã lưu thành công tại: {args.output}")
