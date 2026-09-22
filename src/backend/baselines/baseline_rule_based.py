import argparse
import re
import os
import json

def run_paddleocr(image_path):
    print("Extracting text using PaddleOCR...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4", show_log=False)
    result = ocr.ocr(image_path)
    
    lines = []
    if result and result[0]:
        for line in result[0]:
            lines.append(line[1][0]) # Extract text
    return lines

def run_easyocr(image_path):
    print("Extracting text using EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['vi'], gpu=True)
    result = reader.readtext(image_path)
    
    lines = []
    for line in result:
        lines.append(line[1]) # Extract text
    return lines

def extract_kie_rules(lines):
    print("Applying Rule-based KIE...")
    result = {
        "SELLER": None,
        "ADDRESS": None,
        "TIMESTAMP": None,
        "TOTAL_COST": None,
        "ITEM_NAME": [],
        "ITEM_QTY": [],
        "ITEM_PRICE": [],
        "ITEM_AMOUNT": []
    }
    
    # Heuristic 1: Seller is usually one of the first 2 lines
    if len(lines) > 0:
        result["SELLER"] = lines[0]
        
    # Heuristic 2: Address usually contains keywords like "Đ/c", "Địa chỉ", "Tầng", "Số", "Phường", "Quận"
    address_keywords = ["đ/c", "địa chỉ", "tầng", "số", "phường", "quận", "đường"]
    for line in lines:
        lower_line = line.lower()
        if any(kw in lower_line for kw in address_keywords):
            # Clean up prefix like "Địa chỉ:"
            address = re.sub(r"^(đ/c|địa chỉ)[:\s\-]*", "", lower_line, flags=re.IGNORECASE)
            result["ADDRESS"] = address.title()
            break
            
    # Heuristic 3: Timestamp using Regex (dd/mm/yyyy hh:mm or similar)
    date_pattern = r"(\d{2}[/-]\d{2}[/-]\d{4}\s+\d{2}:\d{2}(:\d{2})?|\d{2}:\d{2}\s+\d{2}[/-]\d{2}[/-]\d{4})"
    for line in lines:
        match = re.search(date_pattern, line)
        if match:
            result["TIMESTAMP"] = match.group(1)
            break
            
    # Heuristic 4: Total Cost using keywords "Tổng tiền", "Total", "Cần thanh toán", "Thanh toán"
    cost_keywords = ["tổng tiền", "tổng cộng", "thanh toán", "cần thanh toán", "tổng thanh toán", "total"]
    for i, line in enumerate(lines):
        lower_line = line.lower()
        if any(kw in lower_line for kw in cost_keywords):
            # Try to find number in the same line
            numbers = re.findall(r"\d{1,3}(?:[.,]\d{3})*", line)
            if numbers:
                result["TOTAL_COST"] = numbers[-1] # Usually the last number is the cost
                break
            elif i + 1 < len(lines):
                # If not in same line, check next line
                next_numbers = re.findall(r"\d{1,3}(?:[.,]\d{3})*", lines[i+1])
                if next_numbers:
                    result["TOTAL_COST"] = next_numbers[-1]
                    break
                    
    return result

def main():
    parser = argparse.ArgumentParser(description="Baseline 1 & 2: Rule-based KIE with OCR")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--ocr", type=str, choices=["paddle", "easyocr"], default="paddle", help="OCR Engine to use")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save the output JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        return
        
    if args.ocr == "paddle":
        lines = run_paddleocr(args.image)
    else:
        lines = run_easyocr(args.image)
        
    print("\n--- RAW OCR TEXT ---")
    for idx, l in enumerate(lines):
        print(f"[{idx}] {l}")
        
    extracted_info = extract_kie_rules(lines)
    
    print("\n--- RULE-BASED EXTRACTION RESULT ---")
    print(json.dumps(extracted_info, ensure_ascii=False, indent=4))
    
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(extracted_info, f, ensure_ascii=False, indent=4)
        print(f"Saved result to {args.output_json}")

if __name__ == "__main__":
    main()
