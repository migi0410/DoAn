import json
import pandas as pd
import os
import random

def create_mcocr_test_jsonl():
    csv_file = "C:/Users/Admin/OneDrive/DoAn/MASTER_DATASET_ARCHIVE/MC_OCR_Dataset/mcocr_train_df.csv"
    img_folder = "C:/Users/Admin/OneDrive/DoAn/MASTER_DATASET_ARCHIVE/MC_OCR_Dataset/train_images/train_images"
    out_file = "C:/Users/Admin/OneDrive/DoAn/test_mcocr.jsonl"
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    samples = []
    
    for idx, row in df.iterrows():
        img_id = row['img_id']
        img_path = os.path.join(img_folder, img_id)
        
        # MCOCR format: anno_texts="A|||B|||C", anno_labels="SELLER|||ADDRESS|||TOTAL_COST"
        texts = str(row['anno_texts']).split('|||')
        labels_list = str(row['anno_labels']).split('|||')
        
        labels_dict = {}
        for text, label in zip(texts, labels_list):
            if label not in labels_dict:
                labels_dict[label] = text
            else:
                labels_dict[label] += " " + text
                
        # Ensure we only pick files that exist
        if os.path.exists(img_path):
            samples.append({
                "id": f"mcocr_{img_id}",
                "image_path": img_path,
                "labels": labels_dict
            })
            
    # Randomly select 500 samples
    random.seed(42)
    if len(samples) > 500:
        samples = random.sample(samples, 500)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Created {out_file} with {len(samples)} samples.")

if __name__ == "__main__":
    create_mcocr_test_jsonl()
