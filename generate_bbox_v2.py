import os
import json

base_dir = r'C:\Users\Admin\OneDrive\DoAn'
runpod_jsonl = os.path.join(base_dir, 'FINAL_RUNPOD_DATASET', 'train.jsonl')
output_json = os.path.join(base_dir, 'FINAL_BBOX_DATASET_V2.json')

# 1. Load hybrid labels (if any exist locally)
hybrid_data = {}
for file in os.listdir(base_dir):
    if file.startswith('hybrid_labels_') and file.endswith('.json'):
        with open(os.path.join(base_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            hybrid_data.update(data)

# 2. Load Label Studio labels from user's Downloads folders
ls_folders = [
    r'C:\Users\Admin\Downloads\Final_label\label',
    r'C:\Users\Admin\Downloads\17_7_label\17_7_label'
]

ls_map = {
    'STORE_NAME': 'SELLER',
    'ADDRESS': 'ADDRESS',
    'DATE': 'TIMESTAMP',
    'TOTAL_AMOUNT': 'TOTAL_COST',
    'ITEM_NAME': 'ITEM_NAME',
    'ITEM_QTY': 'ITEM_QTY',
    'ITEM_PRICE': 'ITEM_PRICE',
    'ITEM_AMOUNT': 'ITEM_AMOUNT'
}

label_studio_data = {}
for folder in ls_folders:
    if not os.path.exists(folder): continue
    for file in os.listdir(folder):
        if file.endswith('.json'):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                tasks = json.load(f)
                for task in tasks:
                    img_path = task.get('data', {}).get('image', '')
                    basename = os.path.basename(img_path)
                    
                    annotations = []
                    # Label studio predictions
                    for pred in task.get('predictions', []):
                        results = pred.get('result', [])
                        
                        # Group by ID
                        elements = {}
                        for r in results:
                            eid = r.get('id')
                            if not eid: continue
                            if eid not in elements:
                                elements[eid] = {}
                            
                            rtype = r.get('type')
                            val = r.get('value', {})
                            
                            orig_w = r.get('original_width', 1000)
                            orig_h = r.get('original_height', 1000)
                            
                            if rtype == 'rectanglelabels':
                                labels = val.get('rectanglelabels', [])
                                if labels:
                                    raw_label = labels[0]
                                    elements[eid]['label'] = ls_map.get(raw_label, raw_label)
                                
                                x = val.get('x', 0)
                                y = val.get('y', 0)
                                w = val.get('width', 0)
                                h = val.get('height', 0)
                                
                                x1 = x * orig_w / 100.0
                                y1 = y * orig_h / 100.0
                                x2 = x1 + (w * orig_w / 100.0)
                                y2 = y1 + (h * orig_h / 100.0)
                                
                                elements[eid]['box'] = [x1, y1, x2, y2]
                                
                            elif rtype == 'textarea':
                                text = val.get('text', [])
                                if text:
                                    elements[eid]['text'] = text[0]
                                    
                        for eid, data in elements.items():
                            if 'label' in data and 'box' in data:
                                annotations.append({
                                    'label': data['label'],
                                    'text': data.get('text', ''),
                                    'box': data['box']
                                })
                                
                    label_studio_data[basename] = annotations

# 3. Load full 10k synthetic labels
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
        
        if img_basename in label_studio_data:
            entry['annotations'] = label_studio_data[img_basename]
        elif img_basename in hybrid_data:
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

print(f'Generated FINAL_BBOX_DATASET_V2.json with {len(final_dataset)} entries.')
print(f'Missing bbox data for {missing_count} images.')
