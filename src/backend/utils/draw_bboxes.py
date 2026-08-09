import json
import io
import cv2
import numpy as np
from PIL import Image
from supabase import create_client
SUPABASE_URL = 'https://gmefyvajylsqsyfpuahk.supabase.co'
SUPABASE_KEY = 'YOUR_SUPABASE_KEY'
BUCKET_NAME = 'raw_images'
JSON_FILE = 'gemini_labels_cafe_starbucks.json'

def draw_bboxes():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for filename, fields in data.items():
        print(f'Drawing for: {filename}')
        remote_path = f'cafe_starbucks/{filename}'
        image_bytes = supabase.storage.from_(BUCKET_NAME).download(remote_path)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        colors = {'SELLER': (0, 0, 255), 'ADDRESS': (0, 255, 0), 'TIMESTAMP': (255, 0, 0), 'TOTAL_COST': (0, 255, 255)}
        for key, field_data in fields.items():
            if not field_data or 'box_2d' not in field_data:
                continue
            box = field_data['box_2d']
            if not box:
                continue
            ymin, xmin, ymax, xmax = box
            ymin_px = int(ymin * h / 1000)
            xmin_px = int(xmin * w / 1000)
            ymax_px = int(ymax * h / 1000)
            xmax_px = int(xmax * w / 1000)
            color = colors.get(key, (255, 255, 255))
            cv2.rectangle(img, (xmin_px, ymin_px), (xmax_px, ymax_px), color, 2)
            cv2.putText(img, key, (xmin_px, ymin_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out_name = f'bbox_drawn_{filename}'
        cv2.imwrite(out_name, img)
        print(f'Saved visualization to {out_name}')
        break
if __name__ == '__main__':
    draw_bboxes()
