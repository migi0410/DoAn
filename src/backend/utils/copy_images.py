import os
import shutil

def copy_images_to_flat_folder(txt_file_path, images_dir, output_dir):
    if not os.path.exists(txt_file_path):
        print(f"Error: Không tìm thấy file danh sách '{txt_file_path}'")
        return
    if not os.path.exists(images_dir):
        print(f"Error: Không tìm thấy thư mục gốc chứa ảnh '{images_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Read image list
    with open(txt_file_path, "r", encoding="utf-8") as f:
        image_names = [line.strip() for line in f if line.strip()]

    print(f"Bắt đầu sao chép {len(image_names)} ảnh vào thư mục '{output_dir}'...")
    
    copied_count = 0
    missing_count = 0
    
    for img_name in image_names:
        img_path = os.path.join(images_dir, img_name)
        dest_path = os.path.join(output_dir, img_name)
        
        if os.path.exists(img_path):
            shutil.copy2(img_path, dest_path)
            copied_count += 1
        else:
            # Recursive search if not found directly
            found = False
            for root, dirs, files in os.walk(images_dir):
                if img_name in files:
                    shutil.copy2(os.path.join(root, img_name), dest_path)
                    copied_count += 1
                    found = True
                    break
            if not found:
                missing_count += 1

    print(f"Sao chép HOÀN TẤT!")
    print(f"- Số lượng ảnh đã sao chép thành công: {copied_count}")
    if missing_count > 0:
        print(f"- CẢNH BÁO: Không tìm thấy {missing_count} ảnh trong thư mục gốc.")
    print(f"- Các ảnh phẳng đã được lưu tại: {os.path.abspath(output_dir)}")
    print(f"Bây giờ nhóm của bạn chỉ cần vào thư mục này, nhấn Ctrl+A chọn tất cả và kéo thả trực tiếp vào Label Studio!")

if __name__ == "__main__":
    train_images_dir = "mcocr/train_images"
    
    # Copy VinCommerce images
    copy_images_to_flat_folder(
        "filtered_templates/vincommerce_winmart_vinmart.txt",
        train_images_dir,
        "vincommerce_flat_images"
    )
    
    # Copy Saigon Co.op images
    copy_images_to_flat_folder(
        "filtered_templates/saigon_coop_coopmart_coop_food.txt",
        train_images_dir,
        "saigon_coop_flat_images"
    )
