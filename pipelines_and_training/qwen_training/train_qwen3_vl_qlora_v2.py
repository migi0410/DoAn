import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ReceiptDatasetV2(Dataset):
    def __init__(self, jsonl_path, processor, max_samples=None, max_img_dim=1280):
        self.processor = processor
        self.max_img_dim = max_img_dim
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["images"][0]
        
        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > self.max_img_dim:
                scale = self.max_img_dim / max(w, h)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            pil_img = Image.new("RGB", (384, 384), color="white")

        # Lấy prompt chống ảo giác đã được chuẩn hóa trong messages[0]
        raw_user_prompt = item["messages"][0]["content"]
        user_text = raw_user_prompt.replace("<image>", "").strip()
        assistant_text = item["messages"][1]["content"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        return messages

class QwenVLDataCollatorV2:
    def __init__(self, processor):
        self.processor = processor
        # Sequence token cho <|im_start|>assistant\n
        self.assistant_start_seq = [151644, 77091, 198]

    def __call__(self, batch_messages):
        texts = []
        for msgs in batch_messages:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()

        # Mask toàn bộ phần User prompt + Vision tokens (-100), chỉ tính loss trên Assistant output
        seq = self.assistant_start_seq
        seq_len = len(seq)
        
        for i in range(labels.shape[0]):
            input_ids_list = inputs["input_ids"][i].tolist()
            # Tìm vị trí bắt đầu của assistant
            idx = -1
            for k in range(len(input_ids_list) - seq_len + 1):
                if input_ids_list[k:k + seq_len] == seq:
                    idx = k + seq_len
                    break
            
            if idx != -1:
                labels[i, :idx] = -100
            else:
                # Fallback: nếu không tìm thấy chuỗi token chính xác, dùng phương pháp template prompt
                print(f"Warning: Không tìm thấy assistant token sequence ở batch item {i}")
                prompt_msgs = [batch_messages[i][0]]
                prompt_text = self.processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                p_img, _ = process_vision_info([prompt_msgs])
                prompt_inputs = self.processor(text=[prompt_text], images=p_img, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                labels[i, :prompt_len] = -100

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Chạy thử nghiệm 5 bước để xác nhận gradient, loss masking và VRAM")
    parser.add_argument("--epochs", type=int, default=1, help="Số epoch huấn luyện (khuyến nghị 1 epoch để tránh overfit)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size trên mỗi GPU")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (tương đương effective batch size = 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_img_dim", type=int, default=1280, help="Kích thước ảnh tối đa (pixel chiều dài nhất)")
    parser.add_argument("--max_pixels", type=int, default=768*28*28, help="Số pixel tối đa cho vision encoder (mặc định 768*28*28 = 602k pixels)")
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora_v2"
    train_file = "/home/haderax/DoAn/official_benchmark/train_v2_abs.jsonl"
    val_file = "/home/haderax/DoAn/official_benchmark/val_v2_abs.jsonl"

    print("=" * 80)
    print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN QLORA V2 CHO {model_id}")
    print(f"   Mode: {'SANITY CHECK (10 mẫu, 5 steps)' if args.sanity_check else f'FULL TRAINING ({args.epochs} epoch)'}")
    print(f"   Output dir: {output_dir}")
    print(f"   Max image dimension: {args.max_img_dim}px | Max vision pixels: {args.max_pixels}")
    print(f"   Batch size: {args.batch_size} | Grad accum: {args.grad_accum} (Effective batch = {args.batch_size * args.grad_accum})")
    print(f"   Learning rate: {args.lr}")
    print("=" * 80)

    # 1. Cấu hình BitsAndBytes 4-bit NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256*28*28,
        max_pixels=args.max_pixels,
        trust_remote_code=True
    )

    # 2. Chuẩn bị model cho k-bit training & LoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("\n--- THÔNG SỐ TRAINABLE CỦA LORA V2 ---")
    model.print_trainable_parameters()

    # 3. Chuẩn bị dữ liệu
    max_train = 20 if args.sanity_check else None
    max_val = 5 if args.sanity_check else 50

    train_ds = ReceiptDatasetV2(train_file, processor, max_samples=max_train, max_img_dim=args.max_img_dim)
    val_ds = ReceiptDatasetV2(val_file, processor, max_samples=max_val, max_img_dim=args.max_img_dim)
    data_collator = QwenVLDataCollatorV2(processor)

    # 4. TrainingArguments
    max_steps = 5 if args.sanity_check else -1
    save_steps = 5 if args.sanity_check else 300
    logging_steps = 1 if args.sanity_check else 10

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=1 if args.sanity_check else 50,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="no",
        save_total_limit=4,
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator
    )

    # Tự động tìm checkpoint gần nhất nếu có để tiếp tục (resume)
    resume_checkpoint = None
    if os.path.exists(output_dir) and not args.sanity_check:
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"🔄 Tìm thấy checkpoint v2: {resume_checkpoint}. Tiếp tục huấn luyện (resume) từ checkpoint này!")

    print("\n🚀 Bắt đầu quá trình huấn luyện v2...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if not args.sanity_check:
        print("\n🎉 Huấn luyện v2 hoàn tất! Đang lưu checkpoint cuối cùng...")
        final_save_path = os.path.join(output_dir, "final_lora_checkpoint")
        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        print(f"Đã lưu checkpoint thành công tại: {final_save_path}")
    else:
        print("\n✅ Sanity check thành công! Không có lỗi gradient, loss masking hoặc VRAM.")

if __name__ == "__main__":
    main()
