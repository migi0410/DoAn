import json

path = 'notebooks/PhoBERT_Kaggle_v2.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Modify pip install
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and cell['source'][0].startswith('!pip install'):
        if 'wandb' not in cell['source'][0]:
            cell['source'][0] = cell['source'][0].strip() + ' wandb\n'
        break

# 2. Add WandB login cell after pip install
wandb_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import wandb\n",
        "wandb.login(key='DÁN_ÐO?N_MÃ_API_KEY_C?A_B?N_VÀO_ÐÂY')"
    ]
}
nb['cells'].insert(3, wandb_cell)

# 3. Add report_to='wandb' in TrainingArguments
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and '%%writefile' in cell['source'][0]:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if 'output_dir="./phobert-avir-kie",' in line:
                new_source.append('        report_to="wandb",\n')
                new_source.append('        run_name="PhoBERT-NER-Run1",\n')
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Patched PhoBERT Notebook with WandB!")
