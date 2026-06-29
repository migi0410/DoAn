import subprocess
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    batch_size = 2000
    num_batches = 5
    out_dir = "../synthetic_dataset_hardcore"
    script = "generate_bulk_invoices.py"
    
    # 2000 / 15 templates = ~133 images per template per batch.
    # To be perfectly safe about numbering: 134 items per batch
    items_per_batch = 134
    
    for i in range(num_batches):
        start_idx = i * items_per_batch + 1
        print(f"=== BATCH {i+1}/{num_batches} | Start Index: {start_idx} ===")
        
        cmd = [
            sys.executable, script,
            "--count", str(batch_size),
            "--output_dir", out_dir,
            "--start_idx", str(start_idx)
        ]
        
        success = False
        while not success:
            try:
                subprocess.run(cmd, check=True)
                print(f"--- Xong Batch {i+1} ---")
                success = True
            except subprocess.CalledProcessError as e:
                print(f"Lỗi ở Batch {i+1}: {e}. Đang chạy lại mẻ này...")
                import time
                time.sleep(5)  # Nghỉ 5s cho RAM xả hết trước khi thử lại

if __name__ == "__main__":
    main()
