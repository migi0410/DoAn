import os
import json
import random

# Đường dẫn file test gốc (1166 dòng)
test_jsonl_path = "FINAL_SPLIT_DATASET/test/test.jsonl"
output_500_path = "test_500.jsonl"

if not os.path.exists(test_jsonl_path):
    print(f"❌ Không tìm thấy {test_jsonl_path}")
else:
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Xáo trộn ngẫu nhiên và bốc 500 tấm
    random.seed(42) # Đặt seed để luôn bốc ra đúng 500 tấm này (nếu có chạy lại)
    random.shuffle(lines)
    selected_500 = lines[:500]
    
    with open(output_500_path, 'w', encoding='utf-8') as f:
        for line in selected_500:
            f.write(line + '\n')
            
    print(f"✅ Đã tạo thành công {output_500_path} chứa {len(selected_500)} hóa đơn ngẫu nhiên!")
