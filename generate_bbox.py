import os
import json

base_dir = r'C:\Users\Admin\OneDrive\DoAn'
runpod_jsonl = os.path.join(base_dir, 'FINAL_RUNPOD_DATASET', 'train.jsonl')
output_json = os.path.join(base_dir, 'FINAL_BBOX_DATASET.json')

hybrid_data = {}
for file in os.listdir(base_dir):
    if file.startswith('hybrid_labels_') and file.endswith('.json'):
        with open(os.path.join(base_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            hybrid_data.update(data)

syn_label_dirs = [
    os.path.join(base_dir, 'MASTER_DATASET_ARCHIVE', 'Synthetic_10K_Dataset', 'train', 'labels'),
    os.path.join(base_dir, 'MASTER_DATASET_ARCHIVE', 'Synthetic_10K_Dataset', 'val', 'labels')
]

syn_data = {}
for d in syn_label_dirs:
    if not os.path.exists(d): continue
    for file in os.listdir(d):
        if file.endswith('.json'):
            base = file.replace('.json', '')
            syn_data[base + '.png'] = os.path.join(d, file)
            syn_data[base + '.jpg'] = os.path.join(d, file)
            syn_data[base + '.jpeg'] = os.path.join(d, file)

final_dataset = []
missing_count = 0
with open(runpod_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        img_rel_path = data.get('images', [''])[0]
        img_basename = os.path.basename(img_rel_path)
        
        entry = {
            'file_name': img_rel_path,
            'annotations': []
        }
        
        if img_basename in hybrid_data:
            annos = hybrid_data[img_basename]
            for label, obj in annos.items():
                if obj and isinstance(obj, dict) and 'box_2d' in obj:
                    entry['annotations'].append({
                        'label': label,
                        'text': obj.get('text', ''),
                        'box': obj.get('box_2d', [])
                    })
        elif img_basename in syn_data:
            with open(syn_data[img_basename], 'r', encoding='utf-8') as sf:
                s_data = json.load(sf)
                for ann in s_data.get('annotations', []):
                    entry['annotations'].append({
                        'label': ann.get('label', ''),
                        'text': ann.get('text', ''),
                        'box': ann.get('box', [])
                    })
        else:
            missing_count += 1
            
        final_dataset.append(entry)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

print(f'Generated FINAL_BBOX_DATASET.json with {len(final_dataset)} entries.')
print(f'Missing bbox data for {missing_count} images.')
