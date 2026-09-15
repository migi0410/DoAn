
import json
import os

ENHANCED_PROMPT = """<image>Bạn là chuyên gia trích xuất thông tin hóa đơn tiếng Việt. Hãy đọc kỹ hình ảnh và trích xuất thông tin vào định dạng JSON sau:
{
  "SELLER": "Tên cửa hàng hoặc công ty",
  "ADDRESS": "Địa chỉ",
  "TIMESTAMP": "Thời gian lập hóa đơn",
  "ITEMS": [
    {
      "name": "Tên món hàng",
      "qty": "Số lượng",
      "price": "Đơn giá",
      "amount": "Thành tiền"
    }
  ],
  "TOTAL_COST": "Tổng tiền thanh toán cuối cùng"
}
Quy tắc bắt buộc:
1. Chỉ trích xuất CHÍNH XÁC những gì nhìn thấy trên ảnh. Giữ nguyên dấu tiếng Việt và chính tả gốc trên hóa đơn. Tuyệt đối KHÔNG tự ý thêm/bớt dấu và KHÔNG bịa đặt thông tin.
2. Với mỗi mặt hàng, nhóm đủ 4 trường: name, qty, price, amount vào cùng một đối tượng.
3. Nếu tên một món hàng dài bị in rớt xuống 2-3 dòng, hãy ghép lại thành một tên món hoàn chỉnh duy nhất.
4. Dòng in số lượng x đơn giá (ví dụ: '1.000 KG x 11.900') hoặc dòng khuyến mãi/giảm giá nằm ngay bên dưới tên món phải được gộp đúng vào món hàng đó, KHÔNG tách thành các món rời rạc.
5. Nếu hóa đơn không in đơn giá riêng (chỉ có số lượng và thành tiền), hãy để price là "".
6. Nếu trường thông tin nào không xuất hiện trên hóa đơn, hãy để giá trị là ""."""

def process_file(in_path, out_path):
    print(f"Processing {in_path} -> {out_path}...")
    count = 0
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            # Replace user prompt
            for m in sample.get("messages", []):
                if m.get("role") == "user":
                    m["content"] = ENHANCED_PROMPT
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    print(f"Successfully wrote {count} samples to {out_path}!")

process_file("/home/haderax/DoAn/official_benchmark/train_v2_abs.jsonl", "/home/haderax/DoAn/official_benchmark/train_v2_enhanced.jsonl")
process_file("/home/haderax/DoAn/official_benchmark/val_v2_abs.jsonl", "/home/haderax/DoAn/official_benchmark/val_v2_enhanced.jsonl")
