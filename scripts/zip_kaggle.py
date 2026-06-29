import os
import zipfile

def zip_for_kaggle():
    output_filename = 'Kaggle_MCOCR_Benchmark.zip'
    base_dir = r'C:\Users\Admin\OneDrive\DoAn'
    
    targets = [
        'pipelines_and_training',
        'utils',
        'trained_models',
        'label_studio_tasks/task_vuong/images',
        'synthetic_dataset_v3/val',
        'Vuong_Label.json'
    ]
    
    print('Bat dau nen...')
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
    zip_for_kaggle()
