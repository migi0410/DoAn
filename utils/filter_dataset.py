import os
import csv
import json
import ast
import re
from collections import Counter

# ---------------------------------------------------------
# DEFINE STORE TEMPLATES AND KEYWORDS (Vietnamese Chains)
# ---------------------------------------------------------
STORE_TEMPLATES = {
    "VinCommerce / WinMart / VinMart": [
        r"VINCOMMERCE", r"VINMART", r"WINMART", r"VCM", r"VIN COM", r"WIN MART"
    ],
    "Saigon Co.op (Co.opmart / Co.op Food)": [
        r"COOPMART", r"CO\.OPMART", r"COOP MART", r"CO\.OP MART", r"SAIGON COOP", r"SAIGON CO\.OP", r"COOP FOOD", r"CO\.OP FOOD"
    ],
    "Bách Hóa Xanh": [
        r"BÁCH HÓA XANH", r"BACH HOA XANH", r"BHX", r"BACHHOAXANH"
    ],
    "Circle K": [
        r"CIRCLE K", r"CIRCLE-K", r"CIRCLEK"
    ],
    "GS25": [
        r"GS25", r"GS 25"
    ],
    "Lotte Mart": [
        r"LOTTE"
    ],
    "Big C / Go!": [
        r"BIG C", r"BIGC", r"GO!", r"GO\s+MALL"
    ],
    "Pharmacity": [
        r"PHARMACITY", r"PHARMA CITY"
    ],
    "Guardian": [
        r"GUARDIAN"
    ],
    "7-Eleven": [
        r"7-ELEVEN", r"7 ELEVEN", r"7ELEVEN"
    ],
    "Highlands Coffee": [
        r"HIGHLANDS", r"HIGHLAND"
    ],
    "Phúc Long": [
        r"PHÚC LONG", r"PHUC LONG"
    ],
    "KFC": [
        r"KFC", r"KENTUCKY"
    ],
    "Jollibee": [
        r"JOLLIBEE"
    ],
    "Trung Nguyên Legend / E-Coffee": [
        r"TRUNG NGUYÊN", r"TRUNG NGUYEN", r"LEGEND"
    ],
    "The Coffee House": [
        r"THE COFFEE HOUSE", r"COFFEE HOUSE"
    ]
}

def parse_literal_list(val):
    """Safely parses a string representation of a list/JSON array."""
    if not val:
        return []
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        try:
            # Try standard JSON parsing
            return json.loads(val)
        except Exception:
            try:
                # Fallback to python literal evaluation
                return ast.literal_eval(val)
            except Exception:
                # Fallback to basic comma split
                content = val[1:-1]
                return [s.strip().strip("'\"") for s in content.split(",") if s.strip()]
    return [val]

def analyze_and_filter_dataset(csv_path, output_dir="filtered_templates"):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' không tồn tại.")
        print("Vui lòng tải file train.csv từ Kaggle về và đặt vào thư mục làm việc, hoặc chỉ định đường dẫn chính xác.")
        return
        
    print(f"Reading and analyzing '{csv_path}'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for filtered image IDs
    filtered_images = {template: [] for template in STORE_TEMPLATES}
    filtered_images["Other / Unmatched"] = []
    
    total_rows = 0
    parse_errors = 0
    
    # Open CSV
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        # Detect delimiter (usually comma or semicolon)
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if ";" in sample else ","
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Verify required columns exist
        headers = reader.fieldnames
        print(f"Headers found in CSV: {headers}")
        
        img_col = next((h for h in headers if "img" in h.lower() or "id" in h.lower()), None)
        text_col = next((h for h in headers if "text" in h.lower()), None)
        if not text_col:
            text_col = next((h for h in headers if "anno" in h.lower()), None)
        
        if not img_col or not text_col:
            print("Error: Không tìm thấy cột chứa hình ảnh hoặc văn bản gán nhãn trong file CSV.")
            return
            
        print(f"Using image column: '{img_col}' and text column: '{text_col}'")
        
        for row in reader:
            total_rows += 1
            img_id = row[img_col].strip()
            raw_texts = row[text_col]
            
            # Parse the text annotations list
            try:
                texts_list = parse_literal_list(raw_texts)
            except Exception:
                texts_list = [raw_texts]
                parse_errors += 1
                
            # Combine all text blocks into a single uppercase string for matching
            combined_text = " ".join([str(t) for t in texts_list]).upper()
            
            # Check matching store template
            matched = False
            for template, patterns in STORE_TEMPLATES.items():
                for pattern in patterns:
                    if re.search(pattern, combined_text):
                        filtered_images[template].append(img_id)
                        matched = True
                        break
                if matched:
                    break
                    
            if not matched:
                filtered_images["Other / Unmatched"].append(img_id)
                
    # ---------------------------------------------------------
    # PRINT RESULTS & WRITE FILTERED TEXT FILES
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" KẾT QUẢ PHÂN TÍCH & PHÂN LOẠI MẪU HÓA ĐƠN")
    print("="*50)
    print(f"Tổng số hóa đơn phân tích: {total_rows}")
    print(f"Số lỗi đọc nhãn: {parse_errors}")
    print("-"*50)
    
    # Sort templates by size
    sorted_templates = sorted(
        filtered_images.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    # Save the statistics to a markdown report
    report_path = "mcocr_statistics_report.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Báo Cáo Thống Kê & Phân Nhóm Hóa Đơn MC-OCR 2021\n\n")
        rf.write(f"*   **Tổng số lượng ảnh hóa đơn:** {total_rows}\n")
        rf.write(f"*   **Thư mục lưu trữ danh sách đã lọc:** `{output_dir}/`\n\n")
        rf.write("| STT | Tên Chuỗi Cửa Hàng (Template) | Số Lượng Ảnh | Tỷ Lệ (%) | Tệp Danh Sách |\n")
        rf.write("| --- | --- | --- | --- | --- |\n")
        
        idx = 1
        for template, img_list in sorted_templates:
            count = len(img_list)
            percentage = (count / total_rows) * 100 if total_rows > 0 else 0
            
            # Print to stdout
            print(f"{idx:2d}. {template:<38} | {count:4d} ảnh ({percentage:5.1f}%)")
            
            # Write to list file if count > 0
            if count > 0:
                safe_name = template.lower().replace(" / ", "_").replace(" (", "_").replace(")", "").replace(".", "").replace(" ", "_")
                txt_filename = f"{safe_name}.txt"
                txt_path = os.path.join(output_dir, txt_filename)
                
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write("\n".join(img_list))
                    
                rf.write(f"| {idx} | **{template}** | {count} | {percentage:.1f}% | [{txt_filename}](file:///{os.path.abspath(txt_path)}) |\n")
            else:
                rf.write(f"| {idx} | {template} | {count} | {percentage:.1f}% | - |\n")
                
            idx += 1
            
    print("="*50)
    print(f"Đã xuất báo cáo thống kê chi tiết tại: {os.path.abspath(report_path)}")
    print(f"Đã tạo các file danh sách ảnh cụ thể tại thư mục: {os.path.abspath(output_dir)}")
    print("Nhóm của bạn có thể đọc các file này để biết chính xác những file ảnh nào thuộc chuỗi cửa hàng nào để tập trung tải lên Label Studio gán nhãn!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lọc và thống kê dataset MC-OCR 2021 theo chuỗi cửa hàng (template)")
    parser.add_argument("--csv", default="train.csv", help="Đường dẫn tới file train.csv của MC-OCR 2021")
    parser.add_argument("--output", default="filtered_templates", help="Thư mục xuất kết quả")
    args = parser.parse_args()
    
    # Fallback to search in common paths if default train.csv is not found
    csv_path = args.csv
    if not os.path.exists(csv_path):
        common_paths = [
            "train.csv",
            "C:/Users/Admin/Downloads/train.csv",
            "C:/Users/Admin/Downloads/vietnamese-receipts-mc-ocr-2021/train.csv"
        ]
        for p in common_paths:
            if os.path.exists(p):
                csv_path = p
                break
                
    analyze_and_filter_dataset(csv_path, args.output)
