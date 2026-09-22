import os
import json
import random
import cv2

base_dir = r'C:\Users\Admin\OneDrive\DoAn'
json_path = os.path.join(base_dir, 'FINAL_BBOX_DATASET_V3.json')
images_dir = os.path.join(base_dir, 'FINAL_RUNPOD_DATASET')

out_dir = r'C:\Users\Admin\.gemini\antigravity\brain\f25cabe5-dc3c-49d8-a24d-00d155d270e2\scratch\bbox_samples'
os.makedirs(out_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Filter for real images (which start with 'z' or have '2026')
real_entries = [e for e in dataset if 'z79' in e['file_name'] or '2026-' in e['file_name']]

random.seed(43) # Changed seed for a new batch
sample_entries = random.sample(real_entries, min(20, len(real_entries)))

drawn_files = []

for i, entry in enumerate(sample_entries):
    img_rel_path = entry['file_name']
    img_abs_path = os.path.join(images_dir, img_rel_path)
    
    if not os.path.exists(img_abs_path):
        continue
        
    img = cv2.imread(img_abs_path)
    if img is None:
        continue
        
    for ann in entry.get('annotations', []):
        box = ann.get('box', [])
        label = ann.get('label', '')
        
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(img, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    out_name = f"sample_batch2_{i:02d}_{os.path.basename(img_rel_path)}"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, img)
    drawn_files.append(out_path)

print(f"Generated {len(drawn_files)} sample images with bounding boxes.")
print(json.dumps(drawn_files))
