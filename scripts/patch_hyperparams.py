import os

files_to_patch = [
    r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2\3_train_layoutlm_v2.py',
    r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2\baseline_phobert_ner_v2.py'
]

for file in files_to_patch:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'warmup_ratio' not in content:
        content = content.replace('save_strategy="epoch",', 
                                  'save_strategy="epoch",\n        warmup_ratio=0.1,\n        weight_decay=0.01,')
        # Change epochs default from 10 to 5 for large dataset
        content = content.replace('epochs = 2 if args.smoke_test else 10', 
                                  'epochs = 2 if args.smoke_test else 5 # Reduced to 5 for 10k dataset')
        
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Patched {os.path.basename(file)}")
