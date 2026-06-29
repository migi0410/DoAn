import os

files = [
    r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2\3_train_layoutlm_v2.py',
    r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2\baseline_phobert_ner_v2.py'
]

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'WANDB_PROJECT' not in content:
        # Add env var at the top
        content = "import os\nos.environ['WANDB_PROJECT'] = 'avir-kie-10k'\n" + content
        
        # Add report_to wandb
        content = content.replace('save_strategy="epoch",', 'save_strategy="epoch",\n        report_to="wandb",')
        
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Patched wandb into {os.path.basename(file)}")
