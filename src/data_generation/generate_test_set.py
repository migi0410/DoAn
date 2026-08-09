import os
import shutil
import json
from generate_bulk_invoices import run_bulk_generator

def generate_synthetic_test_set(count=500, final_dir='synthetic_dataset'):
    temp_dir = 'temp_test_dataset'
    print(f'=== BẮT ĐẦU SINH TẬP TEST GIẢ LẬP: {count} ẢNH ===')
    run_bulk_generator(count=count, output_dir=temp_dir, clean_ratio=0.2, mcocr_mode=True, backgrounds=True, save_pdfs=False)
    test_images_dir = os.path.join(final_dir, 'test', 'images')
    test_labels_dir = os.path.join(final_dir, 'test', 'labels')
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    test_metadata = []
    id_counter = 1
    for split in ['train', 'val']:
        split_dir = os.path.join(temp_dir, split)
        if not os.path.exists(split_dir):
            continue
        images_src = os.path.join(split_dir, 'images')
        labels_src = os.path.join(split_dir, 'labels')
        for f_name in os.listdir(images_src):
            shutil.move(os.path.join(images_src, f_name), os.path.join(test_images_dir, f_name))
        for f_name in os.listdir(labels_src):
            shutil.move(os.path.join(labels_src, f_name), os.path.join(test_labels_dir, f_name))
        meta_path = os.path.join(split_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_list = json.load(f)
            for item in meta_list:
                item['id'] = id_counter
                id_counter += 1
                f_name = item['file_name']
                item['image_path'] = f'test/images/{f_name}'
                item['label_path'] = f"test/labels/{f_name.replace('.png', '.json')}"
                item['pdf_path'] = None
                test_metadata.append(item)
    test_meta_path = os.path.join(final_dir, 'test', 'metadata.json')
    with open(test_meta_path, 'w', encoding='utf-8') as f:
        json.dump(test_metadata, f, ensure_ascii=False, indent=2)
    shutil.rmtree(temp_dir)
    print(f'\n[SUCCESS] Hoàn thành sinh tập test giả lập!')
    print(f"Thư mục tập test lưu tại: {os.path.abspath(os.path.join(final_dir, 'test'))}")
    print(f'Số lượng ảnh: {len(test_metadata)}')
    print(f'Metadata file: {test_meta_path}')
if __name__ == '__main__':
    generate_synthetic_test_set(count=500)
