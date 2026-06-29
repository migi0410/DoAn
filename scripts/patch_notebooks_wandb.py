import json
import os
import glob

base_dir = r"C:\Users\Admin\OneDrive\DoAn\Training_Code_v2"
notebooks = glob.glob(os.path.join(base_dir, "*.ipynb"))

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            new_source = []
            
            for line in source:
                # Add wandb to pip install
                if line.startswith("!pip install") and "wandb" not in line:
                    line = line.replace("\n", " wandb\n")
                    if not line.endswith("\n"):
                        line += " wandb"
                        
                # Add report_to="wandb" to TrainingArguments
                if "metric_for_best_model=" in line and "report_to=" not in line:
                    if line.endswith(",\n"):
                        line = line + '        report_to="wandb",\n'
                    elif line.endswith("\n"):
                        line = line.replace("\n", ",\n") + '        report_to="wandb",\n'
                    elif line.endswith('"\n'):
                        line = line.replace('"\n', '",\n') + '        report_to="wandb",\n'
                    else:
                        line = line + ',\n        report_to="wandb",\n'
                        
                new_source.append(line)
                
            cell['source'] = new_source
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Patched {os.path.basename(nb_path)}")
