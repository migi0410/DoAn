# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Script sinh hàng loạt bộ dữ liệu hóa đơn giả lập tiếng Việt (PDF, PNG, JSON).
      Phân chia đều theo 20 template và chia tỉ lệ 80/20 cho tập train/val.
      Hỗ trợ lồng nền mặt bàn, bóng đổ và chuẩn hóa nhãn MC-OCR.
"""

import os
import sys
import json
import random
import time
import argparse
import cv2
from playwright.sync_api import sync_playwright

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_generation')))

# Import modules từ dự án
from invoice_templates import TEMPLATES_MAP
from receipt_augmenter import ReceiptAugmenter
from synthetic_generator import generate_random_invoice_data, render_html_with_data

def run_bulk_generator(count=2500, output_dir="synthetic_dataset", clean_ratio=0.2, mcocr_mode=True, backgrounds=True, save_pdfs=False, start_idx=1):
    print(f"=== BẮT ĐẦU SINH HÀNG LOẠT: {count} HÓA ĐƠN ===")
    print(f"Thư mục đầu ra: {output_dir}")
    print(f"Tỉ lệ ảnh sạch (clean): {clean_ratio * 100}%")
    print(f"Chế độ MC-OCR: {mcocr_mode}")
    print(f"Giả lập nền mặt bàn: {backgrounds}")
    print(f"Lưu file PDF: {save_pdfs}")
    print(f"Chỉ số bắt đầu (Start Index): {start_idx}")
    
    # Lấy danh sách 15 template đã được chốt (bao gồm minimart_anan)
    templates_keys = [
        "einvoice_viettel", "einvoice_vnpt", "receipt_c45_bb",
        "supermarket_winmart", "supermarket_lotte", "supermarket_bachhoaxanh",
        "convenience_circlek", "convenience_gs25", "convenience_7eleven",
        "cafe_highlands", "cafe_phuclong", "cafe_starbucks",
        "restaurant_kfc", "restaurant_jollibee", "minimart_anan"
    ]
    num_templates = len(templates_keys)
    print(f"Số lượng template hoạt động: {num_templates} ({templates_keys})")
    
    # Chia đều số lượng cho từng template
    count_per_template = count // num_templates
    if count_per_template == 0:
        count_per_template = 1
    print(f"Số lượng ảnh trên mỗi template: {count_per_template}")
    
    # Tạo cấu trúc thư mục cho train/val
    splits = ["train", "val"]
    dirs = {}
    for split in splits:
        dirs[split] = {
            "images": os.path.join(output_dir, split, "images"),
            "labels": os.path.join(output_dir, split, "labels")
        }
        if save_pdfs:
            dirs[split]["pdfs"] = os.path.join(output_dir, split, "pdfs")
        for d in dirs[split].values():
            os.makedirs(d, exist_ok=True)
            
    train_metadata = []
    val_metadata = []
    start_time = time.time()
    
    # Khởi chạy Playwright
    with sync_playwright() as p:
        print("[INFO] Đang khởi động trình duyệt Chromium...")
        browser = p.chromium.launch()
        
        # Mở các context riêng biệt cho các kiểu bố cục để tối ưu tốc độ
        thermal_context = browser.new_context(viewport={"width": 380, "height": 800}, device_scale_factor=1.0)
        a4_context = browser.new_context(viewport={"width": 794, "height": 1123}, device_scale_factor=1.0)
        c45_context = browser.new_context(viewport={"width": 650, "height": 450}, device_scale_factor=1.0)
        
        thermal_page = thermal_context.new_page()
        a4_page = a4_context.new_page()
        c45_page = c45_context.new_page()
        
        total_generated = 0
        
        # Lặp qua từng template để chia đều số lượng
        for t_idx, template_key in enumerate(templates_keys):
            print(f"\n[INFO] Đang sinh cho template: {template_key} ({t_idx+1}/{num_templates})")
            
            # Chia 80% train, 20% val cho mỗi template để đảm bảo stratified split
            num_train = int(count_per_template * 0.8)
            num_val = count_per_template - num_train
            
            runs = [("train", j) for j in range(start_idx, start_idx + num_train)] + [("val", j) for j in range(start_idx, start_idx + num_val)]
            
            for split, run_idx in runs:
                total_generated += 1
                prefix = f"{template_key}_{split}_{run_idx:03d}"
                
                # Sinh dữ liệu ngẫu nhiên phù hợp với template
                invoice_data = generate_random_invoice_data(template_key)
                
                is_a4 = template_key.startswith("einvoice_")
                is_c45 = (template_key == "receipt_c45_bb")
                
                if is_a4:
                    page = a4_page
                    viewport_width = 794
                elif is_c45:
                    page = c45_page
                    viewport_width = 650
                else:
                    page = thermal_page
                    viewport_width = 380
                    
                # 2. Render HTML
                html_content = render_html_with_data(invoice_data, template_key)
                page.set_content(html_content)
                page.wait_for_timeout(200) # Đợi font vẽ xong
                
                # Thư mục đích dựa trên split
                dest_dirs = dirs[split]
                temp_png_path = os.path.join(dest_dirs["images"], f"{prefix}.png")
                json_path = os.path.join(dest_dirs["labels"], f"{prefix}.json")
                
                # Screenshot
                page.screenshot(path=temp_png_path, full_page=True)
                
                # Lưu PDF
                if save_pdfs:
                    pdf_path = os.path.join(dest_dirs["pdfs"], f"{prefix}.pdf")
                    if is_a4:
                        page.pdf(path=pdf_path, format="A4", margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"})
                    elif is_c45:
                        page.pdf(path=pdf_path, width="170mm", height="110mm", margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"})
                    else:
                        page.pdf(path=pdf_path, width="80mm", height="200mm", margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"})
                    
                # 4. Trích xuất tọa độ từ DOM
                box_data = page.evaluate("""() => {
                    const elements = document.querySelectorAll('.kie-field');
                    const result = [];
                    elements.forEach(el => {
                        const rect = el.getBoundingClientRect();
                        result.push({
                            label: el.getAttribute('data-label'),
                            text: el.innerText.trim(),
                            box: [
                                Math.round(rect.left),
                                Math.round(rect.top),
                                Math.round(rect.right),
                                Math.round(rect.bottom)
                            ]
                        });
                    });
                    return result;
                }""")
                
                # 5. Thiết lập tham số Augmentation ngẫu nhiên
                perspective = 0.0
                shadow = 0.0
                flash = 0.0
                fold = 0.0
                fade = 0.0
                streak = 0.0
                bg_type = "white"
                
                roll = random.random()
                if roll > clean_ratio:
                    # Ảnh bẩn -> Áp dụng ngẫu nhiên các loại nhiễu
                    if roll < clean_ratio + (1 - clean_ratio) / 2:
                        # Mức độ nhiễu khá cao
                        perspective = random.uniform(0.02, 0.06)
                        shadow = random.uniform(0.2, 0.5)
                        flash = random.uniform(0.2, 0.5)
                        fold = random.uniform(1.0, 3.0)
                        fade = random.uniform(0.0, 0.2)
                    else:
                        # Mức độ nhiễu "Hardcore"
                        perspective = random.uniform(0.05, 0.1)
                        shadow = random.uniform(0.4, 0.75)
                        flash = random.uniform(0.4, 0.75)
                        fold = random.uniform(3.0, 8.0)
                        fade = random.uniform(0.2, 0.5)
                        if not is_a4 and not is_c45:
                            streak = random.uniform(0.3, 0.6)
                            
                    # Bật giả lập nền mặt bàn ngẫu nhiên
                    if backgrounds:
                        bg_type = random.choice(["wood", "concrete", "desk"])
                        
                # 6. Chạy bộ tăng cường ảnh bằng OpenCV
                if perspective > 0 or shadow > 0 or flash > 0 or fold > 0 or fade > 0 or streak > 0 or bg_type != "white":
                    try:
                        img_augmented, box_data_augmented = ReceiptAugmenter.apply_pipeline(
                            temp_png_path, 
                            box_data, 
                            perspective_level=perspective,
                            shadow_level=shadow,
                            flash_level=flash,
                            fold_level=fold,
                            fade_level=fade,
                            streak_level=streak,
                            bg_type=bg_type
                        )
                        cv2.imwrite(temp_png_path, img_augmented)
                        box_data = box_data_augmented
                    except Exception as e:
                        print(f"\n[WARNING] Lỗi lưu text cho {prefix}: {str(e)}")
                    
                # 7. Ánh xạ nhãn sang 4 nhãn chuẩn của MC-OCR
                if mcocr_mode:
                    mapped_box_data = []
                    label_map = {
                        "store_name": "SELLER",
                        "address": "ADDRESS",
                        "date": "TIMESTAMP",
                        "total_amount": "TOTAL_COST"
                    }
                    for ann in box_data:
                        orig_label = str(ann["label"])
                        if orig_label in label_map:
                            new_ann = ann.copy()
                            new_ann["label"] = label_map[orig_label]
                            mapped_box_data.append(new_ann)
                        elif orig_label.startswith("item_name"):
                            new_ann = ann.copy()
                            new_ann["label"] = "ITEM_NAME"
                            mapped_box_data.append(new_ann)
                        elif orig_label.startswith("item_qty"):
                            new_ann = ann.copy()
                            new_ann["label"] = "ITEM_QTY"
                            mapped_box_data.append(new_ann)
                        elif orig_label.startswith("item_price"):
                            new_ann = ann.copy()
                            new_ann["label"] = "ITEM_PRICE"
                            mapped_box_data.append(new_ann)
                        elif orig_label.startswith("item_amount"):
                            new_ann = ann.copy()
                            new_ann["label"] = "ITEM_AMOUNT"
                            mapped_box_data.append(new_ann)
                    box_data = mapped_box_data
                    
                image_height = page.evaluate("() => document.body.scrollHeight")
                
                # 8. Lưu nhãn JSON
                label_data = {
                    "file_name": f"{prefix}.png",
                    "image_width": viewport_width,
                    "image_height": image_height,
                    "annotations": box_data
                }
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                    
                # 9. Lưu vào danh sách metadata của split tương ứng
                meta_item = {
                    "id": total_generated,
                    "file_name": f"{prefix}.png",
                    "template": template_key,
                    "store_name": invoice_data["store_name"],
                    "total_amount": invoice_data["total_amount"],
                    "image_path": f"{split}/images/{prefix}.png",
                    "label_path": f"{split}/labels/{prefix}.json",
                    "pdf_path": f"{split}/pdfs/{prefix}.pdf" if save_pdfs else None,
                    "is_a4": is_a4,
                    "is_c45": is_c45,
                    "augmentations": {
                        "perspective": round(perspective, 3),
                        "shadow": round(shadow, 3),
                        "flash": round(flash, 3),
                        "fold": round(fold, 3),
                        "fade": round(fade, 3),
                        "streak": round(streak, 3),
                        "background": bg_type
                    }
                }
                
                if split == "train":
                    train_metadata.append(meta_item)
                else:
                    val_metadata.append(meta_item)
                    
                # In tiến trình
                if total_generated % 50 == 0 or total_generated == count:
                    elapsed = time.time() - start_time
                    avg_speed = elapsed / total_generated
                    eta = avg_speed * (count - total_generated)
                    print(f"[PROGRESS] Đã sinh {total_generated}/{count} hóa đơn ({total_generated/count*100:.1f}%) | Tốc độ: {avg_speed:.2f}s/ảnh | ETA: {eta/60:.1f} phút")
                    
        # Đóng các context và trình duyệt
        thermal_context.close()
        a4_context.close()
        c45_context.close()
        browser.close()
        
    # Ghi file metadata cho từng tập
    with open(os.path.join(output_dir, "train", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "val", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(val_metadata, f, ensure_ascii=False, indent=2)
        
    # Ghi file metadata tổng hợp ở thư mục gốc
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(train_metadata + val_metadata, f, ensure_ascii=False, indent=2)
        
    total_elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Hoàn thành sinh bộ dữ liệu {total_generated} hóa đơn!")
    print(f"Tập Train: {len(train_metadata)} ảnh | Tập Val: {len(val_metadata)} ảnh")
    print(f"Tổng thời gian: {total_elapsed/60:.2f} phút")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu hóa đơn tiếng Việt hàng loạt (MC-OCR)")
    parser.add_argument("--count", type=int, default=2500, help="Tổng số lượng hóa đơn cần sinh cho cả train và val")
    parser.add_argument("--output_dir", default="synthetic_dataset", help="Thư mục đầu ra")
    parser.add_argument("--clean_ratio", type=float, default=0.2, help="Tỉ lệ hóa đơn sạch hoàn toàn (0.0 đến 1.0)")
    parser.add_argument("--start_idx", type=int, default=1, help="Chỉ số bắt đầu cho việc đặt tên file")
    parser.add_argument("--no_mcocr", action="store_true", help="Tắt chế độ lọc và ánh xạ nhãn MC-OCR")
    parser.add_argument("--no_backgrounds", action="store_true", help="Tắt giả lập nền mặt bàn và bóng đổ")
    parser.add_argument("--save_pdfs", action="store_true", help="Lưu thêm file PDF sạch bên cạnh ảnh")
    
    args = parser.parse_args()
    
    mcocr_mode = not args.no_mcocr
    backgrounds = not args.no_backgrounds
    
    run_bulk_generator(
        count=args.count,
        output_dir=args.output_dir,
        clean_ratio=args.clean_ratio,
        mcocr_mode=mcocr_mode,
        backgrounds=backgrounds,
        save_pdfs=args.save_pdfs,
        start_idx=args.start_idx
    )
