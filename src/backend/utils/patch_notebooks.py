import json
import re

def patch_notebook(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'def load_data_from_metadata' in ''.join(cell['source']):
            source = ''.join(cell['source'])
            
            # Replace LABELS
            old_labels_regex = r'LABELS\s*=\s*\[\s*"O",.*?"I-OTHER"\s*\]'
            new_labels = '''LABELS = [
    "O", "B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS",
    "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST",
    "B-ITEM_NAME", "I-ITEM_NAME", "B-ITEM_QTY", "I-ITEM_QTY",
    "B-ITEM_PRICE", "I-ITEM_PRICE", "B-ITEM_AMOUNT", "I-ITEM_AMOUNT",
    "B-OTHER", "I-OTHER"
]'''
            if re.search(old_labels_regex, source, flags=re.DOTALL):
                source = re.sub(old_labels_regex, new_labels, source, flags=re.DOTALL)
            else:
                print('old_labels not found in', filename)

            # Replace logic
            old_logic_regex = r'if label not in \["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"\]:\s*label = "OTHER"'
            new_logic = '''label = str(ann['label']).upper()
            if label.startswith("ITEM_NAME"): label = "ITEM_NAME"
            elif label.startswith("ITEM_QTY"): label = "ITEM_QTY"
            elif label.startswith("ITEM_PRICE"): label = "ITEM_PRICE"
            elif label.startswith("ITEM_AMOUNT"): label = "ITEM_AMOUNT"
            elif label not in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]:
                label = "OTHER"'''
            if re.search(old_logic_regex, source):
                source = re.sub(old_logic_regex, new_logic, source)
            else:
                print('old_logic not found in', filename)
            
            # Put back
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print('Updated', filename)

patch_notebook('LayoutLMv3_Kaggle_v2.ipynb')
patch_notebook('LayoutLM_Kaggle_v2.ipynb')
