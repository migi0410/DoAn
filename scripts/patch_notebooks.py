import os
import json

base_dir = r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2'
notebooks = ['LayoutLM_Kaggle_v2.ipynb', 'LayoutLMv3_Kaggle_v2.ipynb', 'PhoBERT_Kaggle_v2.ipynb']

for nb in notebooks:
    path = os.path.join(base_dir, nb)
    if not os.path.exists(path):
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('/kaggle/working/synthetic_dataset', '/kaggle/working/synthetic_dataset_hardcore')
    content = content.replace('synthetic_dataset_hardcore_hardcore', 'synthetic_dataset_hardcore')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Patched properly {nb}')
