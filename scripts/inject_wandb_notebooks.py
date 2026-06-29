import os
import json

base_dir = r'C:\Users\Admin\OneDrive\DoAn\Training_Code_v2'
notebooks = ['LayoutLM_Kaggle_v2.ipynb', 'LayoutLMv3_Kaggle_v2.ipynb', 'PhoBERT_Kaggle_v2.ipynb']

wandb_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- CẤU HÌNH WANDB (Vẽ biểu đồ Loss/F1) ---\n",
        "import os\n",
        "# Lấy mã API tại: https://wandb.ai/authorize\n",
        "os.environ['WANDB_API_KEY'] = 'DÁN_MÃ_API_CỦA_BẠN_VÀO_ĐÂY'\n"
    ]
}

for nb in notebooks:
    path = os.path.join(base_dir, nb)
    if not os.path.exists(path):
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if we already injected it
    first_cell_source = "".join(data['cells'][0].get('source', []))
    second_cell_source = "".join(data['cells'][1].get('source', [])) if len(data['cells']) > 1 else ""
    
    if 'WANDB_API_KEY' not in first_cell_source and 'WANDB_API_KEY' not in second_cell_source:
        # Insert after the first markdown cell, or at the beginning if no markdown
        if data['cells'][0]['cell_type'] == 'markdown':
            data['cells'].insert(1, wandb_cell)
        else:
            data['cells'].insert(0, wandb_cell)
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'Injected WandB cell into {nb}')
    else:
        print(f'WandB cell already exists in {nb}')
