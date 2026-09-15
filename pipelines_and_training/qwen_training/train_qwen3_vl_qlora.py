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

class ReceiptDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_samples=None, max_img_dim=768):
        self.processor = processor
        self.max_img_dim = max_img_dim
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
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

        user_text = "Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON."
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

class QwenVLDataCollator:
    def __init__(self, processor):
        self.processor = processor

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

        # Mask nhãn prompt user (-100)
        for i, msgs in enumerate(batch_messages):
            prompt_msgs = [msgs[0]]
            prompt_text = self.processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.processor(text=[prompt_text], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Chạy thử nghiệm 10 mẫu để xác nhận tính toán gradient và VRAM")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size trên mỗi GPU")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir = "/home/haderax/DoAn/checkpoints_qwen3_vl_qlora"
    train_file = "/home/haderax/DoAn/official_benchmark/train_abs.jsonl"
    val_file = "/home/haderax/DoAn/official_benchmark/val_abs.jsonl"

    print("=" * 80)
    print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN QLORA CHO {model_id}")
    print(f"   Mode: {'SANITY CHECK (10 mẫu)' if args.sanity_check else 'FULL TRAINING (9,322 mẫu)'}")
    print(f"   Output dir: {output_dir}")
    print("=" * 80)

    # 1. Cấu hình BitsAndBytes 4-bit tối ưu
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
    
    # Giới hạn max_pixels ở mức 512*28*28 (~400K pixels - hoàn hảo cho hóa đơn A4 / bill nhiệt)
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256*28*28,
        max_pixels=512*28*28,
        trust_remote_code=True
    )

    # 2. Chuẩn bị model cho k-bit training & LoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Nhắm vào Attention modules (q, k, v, o)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("\n--- THÔNG SỐ TRAINABLE CỦA LORA ---")
    model.print_trainable_parameters()

    # 3. Chuẩn bị dữ liệu
    max_train = 10 if args.sanity_check else None
    max_val = 5 if args.sanity_check else 50

    train_ds = ReceiptDataset(train_file, processor, max_samples=max_train, max_img_dim=768)
    val_ds = ReceiptDataset(val_file, processor, max_samples=max_val, max_img_dim=768)
    data_collator = QwenVLDataCollator(processor)

    # 4. TrainingArguments
    max_steps = 10 if args.sanity_check else -1
    save_steps = 5 if args.sanity_check else 500
    eval_steps = 5 if args.sanity_check else 500
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
        save_total_limit=3,
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
    if os.path.exists(output_dir):
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"🔄 Tìm thấy checkpoint: {resume_checkpoint}. Tiếp tục huấn luyện (resume) từ checkpoint này!")

    print("\n🚀 Bắt đầu quá trình huấn luyện...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\n🎉 Huấn luyện hoàn tất! Đang lưu checkpoint cuối cùng...")
    final_save_path = os.path.join(output_dir, "final_lora_checkpoint")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Đã lưu checkpoint thành công tại: {final_save_path}")

if __name__ == "__main__":
    main()
