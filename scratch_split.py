import json
import os
import random
import shutil

random.seed(42)

base_dir = r'C:\Users\Admin\OneDrive\DoAn\FINAL_RUNPOD_DATASET'
jsonl_path = os.path.join(base_dir, 'train.jsonl')
output_base = r'C:\Users\Admin\OneDrive\DoAn\FINAL_SPLIT_DATASET'

print('Reading data...')
data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

n = len(data)
# Split ratio: 80% Train, 10% Val, 10% Test
train_end = int(n * 0.8)
val_end = int(n * 0.9)

splits = {
    'train': data[:train_end],
    'val': data[train_end:val_end],
    'test': data[val_end:]
}

os.makedirs(output_base, exist_ok=True)

print(f'Starting split into {output_base} ...')
for split_name, split_data in splits.items():
    split_dir = os.path.join(output_base, split_name)
    img_dir = os.path.join(split_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    out_jsonl = os.path.join(split_dir, f'{split_name}.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for item in split_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            for img_rel_path in item.get('images', []):
                src_path = os.path.join(base_dir, img_rel_path)
                dst_path = os.path.join(split_dir, img_rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    try:
                        # Try hardlink first to save disk space and time
                        os.link(src_path, dst_path)
                    except OSError:
                        # Fallback to copy
                        shutil.copy2(src_path, dst_path)

print(f"Split complete! Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
