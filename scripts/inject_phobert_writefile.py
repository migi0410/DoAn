import json
import os

base_dir = r"C:\Users\Admin\OneDrive\DoAn\Training_Code_v2"
notebook_path = os.path.join(base_dir, "PhoBERT_Kaggle_v2.ipynb")
script_path = os.path.join(base_dir, "baseline_phobert_ner_v2.py")

with open(script_path, 'r', encoding='utf-8') as f:
    script_lines = f.readlines()
    
# Create the source for the cell
source = ["%%writefile /kaggle/working/baseline_phobert_ner_v2.py\n"] + script_lines

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source
}

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# Check if we already injected it
has_writefile = False
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        if len(cell.get('source', [])) > 0 and cell['source'][0].startswith("%%writefile"):
            has_writefile = True
            break
            
if not has_writefile:
    # Insert it before the cell that runs !python
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code' and len(cell.get('source', [])) > 0:
            if '!python' in "".join(cell['source']):
                data['cells'].insert(i, new_cell)
                break
                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Successfully injected %%writefile into PhoBERT_Kaggle_v2.ipynb")
else:
    print("Already has %%writefile")
