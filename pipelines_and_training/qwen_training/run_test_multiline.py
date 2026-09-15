
import io, re, json, time, torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

img_path = '/home/haderax/DoAn/test_winmart_user.jpg'
image = Image.open(img_path).convert('RGB')
w, h = image.size

PROMPT_MULTILINE = '''Bạn là chuyên gia trích xuất thông tin hóa đơn tiếng Việt. Hãy đọc kỹ hình ảnh và trích xuất thông tin vào định dạng JSON sau:
{
  "SELLER": "Tên cửa hàng hoặc công ty",
  "ADDRESS": "Địa chỉ",
  "TIMESTAMP": "Thời gian lập hóa đơn",
  "ITEMS": [
    {
      "name": "Tên món hàng (gộp đủ các dòng)",
      "qty": "Số lượng",
      "price": "Đơn giá",
      "amount": "Thành tiền"
    }
  ],
  "TOTAL_COST": "Tổng tiền thanh toán cuối cùng"
}
Quy tắc đặc biệt quan trọng:
1. Trên hóa đơn này, tên một món hàng được in ngắt thành 2-3 dòng chữ. Bạn BẮT BUỘC phải gộp toàn bộ các dòng của cùng một món thành MỘT tên món duy nhất:
   - Dòng 'NAM DƯƠNG Sốt' + 'Dầu Dấm Trộn' + 'Salad 250g' gộp thành món 'NAM DƯƠNG Sốt Dầu Dấm Trộn Salad 250g' (Đơn giá: 20,200, Thành tiền: 20,200).
   - Dòng 'MỘC CHÂU Sữa' + 'thanh trùng' + 'k.đường H 900ml' gộp thành món 'MỘC CHÂU Sữa thanh trùng k.đường H 900ml' (Đơn giá: 40,700, Thành tiền: 40,700).
   - Dòng 'WINECO Xà lách' + 'lolo xanh L1 300g' gộp thành món 'WINECO Xà lách lolo xanh L1 300g' (Đơn giá: 15,500, Thành tiền: 15,500).
2. Hóa đơn này chỉ có ĐÚNG 3 MÓN HÀNG tương ứng 3 dòng giá tiền. Mảng ITEMS chỉ được có đúng 3 phần tử. Tuyệt đối không tách rời các dòng chữ thành các món riêng biệt khuyết giá tiền.
3. Trích xuất đúng số tiền thanh toán cuối cùng vào TOTAL_COST.'''

# Load model
model_id = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_DIR_V2 = '/home/haderax/DoAn/checkpoints_qwen3_vl_qlora_v2/final_lora_checkpoint'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR_V2)
model.eval()

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256*28*28,
    max_pixels=1024*28*28,
    trust_remote_code=True
)

messages = [
    {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT_MULTILINE}]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

out_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print('=== GENERATED OUTPUT WITH MULTILINE PROMPT ===')
print(out_text)
