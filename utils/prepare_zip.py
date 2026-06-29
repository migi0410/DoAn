import os
import zipfile
import shutil

def create_zip_for_template(template_name, txt_file_path, images_dir, output_zip_path):
    if not os.path.exists(txt_file_path):
        print(f"Error: Không tìm thấy file danh sách '{txt_file_path}'")
        return
    if not os.path.exists(images_dir):
        print(f"Error: Không tìm thấy thư mục chứa ảnh '{images_dir}'")
        return

    # Read image list
    with open(txt_file_path, "r", encoding="utf-8") as f:
        image_names = [line.strip() for line in f if line.strip()]

    print(f"Bắt đầu đóng gói {len(image_names)} ảnh cho nhóm '{template_name}'...")
    
    missing_count = 0
    zipped_count = 0
    
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for img_name in image_names:
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                # Write to zip with only the filename (no directory path inside the zip)
                zipf.write(img_path, img_name)
                zipped_count += 1
            else:
                # Sometimes images are in subdirectories or need path checking
                # Let's search recursively if not found immediately
                found = False
                for root, dirs, files in os.walk(images_dir):
                    if img_name in files:
                        zipf.write(os.path.join(root, img_name), img_name)
                        zipped_count += 1
                        found = True
                        break
                if not found:
                    missing_count += 1

    print(f"Đóng gói HOÀN TẤT!")
    print(f"- Số lượng ảnh đã nén thành công: {zipped_count}")
    if missing_count > 0:
        print(f"- CẢNH BÁO: Không tìm thấy {missing_count} ảnh trong thư mục gốc.")
    print(f"- Tệp ZIP đã lưu tại: {os.path.abspath(output_zip_path)}")
    print(f"- Dung lượng file ZIP: {os.path.getsize(output_zip_path) / (1024*1024):.2f} MB")

def zip_all_images(images_dir, output_zip_path):
    if not os.path.exists(images_dir):
        print(f"Error: Không tìm thấy thư mục chứa ảnh '{images_dir}'")
        return

    print(f"Bắt đầu tìm kiếm và đóng gói TOÀN BỘ ảnh từ '{images_dir}'...")
    
    zipped_count = 0
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    zipf.write(img_path, file)
                    zipped_count += 1

    print(f"Đóng gói HOÀN TẤT!")
    print(f"- Số lượng ảnh đã nén thành công: {zipped_count}")
    print(f"- Tệp ZIP đã lưu tại: {os.path.abspath(output_zip_path)}")
    print(f"- Dung lượng file ZIP: {os.path.getsize(output_zip_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    train_images_dir = "mcocr/train_images"
    
    # 1. Zip VinCommerce (355 ảnh)
    vincommerce_txt = "filtered_templates/vincommerce_winmart_vinmart.txt"
    create_zip_for_template("VinCommerce", vincommerce_txt, train_images_dir, "vincommerce_images.zip")
    
    # 2. Zip Saigon Co.op (41 ảnh)
    coop_txt = "filtered_templates/saigon_coop_coopmart_coop_food.txt"
    create_zip_for_template("Saigon Co.op", coop_txt, train_images_dir, "saigon_coop_images.zip")
    
    # 3. Zip TOÀN BỘ 1155 ảnh (Tùy chọn nếu muốn up hết)
    zip_all_images(train_images_dir, "mcocr_all_images.zip")
