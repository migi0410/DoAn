import os
import zipfile

def zip_for_kaggle_train():
    output_filename = 'Kaggle_Training_10k_Dataset.zip'
    base_dir = r'C:\Users\Admin\OneDrive\DoAn'
    
    targets = [
        'synthetic_dataset_hardcore'
    ]
    
    print('Bat dau nen chi rieng dataset 10k...')
    with zipfile.ZipFile(os.path.join(base_dir, output_filename), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zipf:
        for target in targets:
            target_path = os.path.join(base_dir, target)
            if os.path.isfile(target_path):
                zipf.write(target_path, arcname=target)
            elif os.path.isdir(target_path):
                for root, dirs, files in os.walk(target_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_dir)
                        zipf.write(file_path, arcname=arcname)
                
    print('Hoan thanh!')

if __name__ == '__main__':
    zip_for_kaggle_train()
