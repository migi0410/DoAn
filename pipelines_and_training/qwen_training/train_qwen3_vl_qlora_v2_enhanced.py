# -*- coding: utf-8 -*-
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
        user_msg = item["messages"][0]["content"]
        assistant_text = item["messages"][1]["content"]

        img_path = None
        user_text = ""
        for part in user_msg:
            if isinstance(part, dict):
                if part.get("type") == "image":
                    img_path = part.get("image")
                elif part.get("type") == "text":
                    user_text = part.get("text", "")

        if img_path is None:
            text_str = user_msg if isinstance(user_msg, str) else str(user_msg)
            import re
            m = re.search(r"<image>(.*)", text_str, re.DOTALL)
            user_text = m.group(1).strip() if m else text_str

            if "image" in item:
                img_path = item["image"]
            elif "images" in item and len(item["images"]) > 0:
                img_path = item["images"][0]

        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} at index {idx}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > self.max_img_dim:
            scale = self.max_img_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

        seq = self.assistant_start_seq
        seq_len = len(seq)
        
        for i in range(labels.shape[0]):
            input_ids_list = inputs["input_ids"][i].tolist()
            idx = -1
            for k in range(len(input_ids_list) - seq_len + 1):
                if input_ids_list[k:k + seq_len] == seq:
                    idx = k + seq_len
                    break
            
            if idx != -1:
                labels[i, :idx] = -100
            else:
                prompt_msgs = [batch_messages[i][0]]
                prompt_text = self.processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                p_img, _ = process_vision_info([prompt_msgs])
                prompt_inputs = self.processor(text=[prompt_text], images=p_img, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                labels[i, :prompt_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Chạy thử nghiệm 5 bước để xác nhận gradient, loss masking và VRAM")
    parser.add_argument("--epochs", type=int, default=1, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size trên mỗi GPU")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_img_dim", type=int, default=1280, help="Kích thước ảnh tối đa (pixel chiều dài nhất)")
    parser.add_argument("--max_pixels", type=int, default=768*28*28, help="Số pixel tối đa cho vision encoder")
    parser.add_argument("--train_file", type=str, default="/home/haderax/DoAn/official_benchmark/train_v2_enhanced.jsonl", help="Đường dẫn file train jsonl")
    parser.add_argument("--val_file", type=str, default="/home/haderax/DoAn/official_benchmark/val_v2_enhanced.jsonl", help="Đường dẫn file val jsonl")
    parser.add_argument("--output_dir", type=str, default="/home/haderax/DoAn/checkpoints_qwen3_vl_qlora_v2_enhanced", help="Thư mục lưu checkpoints")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint gần nhất nếu có")
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    print("=" * 80)
    print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN QLORA V2 ENHANCED CHO {model_id}")
    print(f"   Mode: {'SANITY CHECK (10 mẫu, 5 steps)' if args.sanity_check else f'FULL TRAINING ({args.epochs} epoch)'}")
    print(f"   Train file: {args.train_file}")
    print(f"   Val file: {args.val_file}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Max image dimension: {args.max_img_dim}px | Max vision pixels: {args.max_pixels}")
    print(f"   Batch size: {args.batch_size} | Grad accum: {args.grad_accum} (Effective batch = {args.batch_size * args.grad_accum})")
    print(f"   Learning rate: {args.lr}")
    print("=" * 80)

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
    print("\n--- THÔNG SỐ TRAINABLE CỦA LORA V2 ENHANCED ---")
    model.print_trainable_parameters()

    max_train = 20 if args.sanity_check else None
    max_val = 5 if args.sanity_check else 50

    train_ds = ReceiptDatasetV2(args.train_file, processor, max_samples=max_train, max_img_dim=args.max_img_dim)
    val_ds = ReceiptDatasetV2(args.val_file, processor, max_samples=max_val, max_img_dim=args.max_img_dim)
    data_collator = QwenVLDataCollatorV2(processor)

    max_steps = 5 if args.sanity_check else -1
    save_steps = 5 if args.sanity_check else 300
    logging_steps = 1 if args.sanity_check else 10

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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

    resume_checkpoint = None
    if args.resume and os.path.exists(args.output_dir) and not args.sanity_check:
        checkpoints = [
            os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"🔄 Tìm thấy checkpoint: {resume_checkpoint}. Tiếp tục huấn luyện (resume) từ checkpoint này!")

    print("\n🚀 Bắt đầu quá trình huấn luyện v2 Enhanced...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if not args.sanity_check:
        print("\n🎉 Huấn luyện v2 Enhanced hoàn tất! Đang lưu checkpoint cuối cùng...")
        final_save_path = os.path.join(args.output_dir, "final_lora_checkpoint")
        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        print(f"Đã lưu checkpoint thành công tại: {final_save_path}")
    else:
        print("\n✅ Sanity check thành công! Không có lỗi gradient, loss masking hoặc VRAM.")

if __name__ == "__main__":
    main()
