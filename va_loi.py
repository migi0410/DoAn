import os
import importlib.util

# Tự động dò tìm đường dẫn thư viện transformers trên Windows
spec = importlib.util.find_spec("transformers")
if spec is None:
    print("❌ Chưa cài đặt thư viện transformers. Hãy chạy 'pip install transformers' trước!")
    exit()

transformers_dir = os.path.dirname(spec.origin)
print(f"🛠️ Đang tiến hành vá lỗi Transformers tại: {transformers_dir} ...")

# 1. Vá lỗi skip_tensor_conversion
path1 = os.path.join(transformers_dir, "feature_extraction_utils.py")
if os.path.exists(path1):
    with open(path1, 'r', encoding='utf-8') as f: c1 = f.read()
    c1 = c1.replace(', skip_tensor_conversion=skip_tensor_conversion', '')
    with open(path1, 'w', encoding='utf-8') as f: f.write(c1)
    print("✅ Đã vá feature_extraction_utils.py")

# 2. Cấy ghép Getter/Setter an toàn
path2 = os.path.join(transformers_dir, "modeling_utils.py")
if os.path.exists(path2):
    with open(path2, 'a', encoding='utf-8') as f:
        f.write('\n\n# ==========================================\n')
        f.write('# MINI CPM FIX: INJECT SAFE PROPERTY WITH SETTER\n')
        f.write('# ==========================================\n')
        f.write('def _safe_tied_keys_getter(self):\n')
        f.write('    res = getattr(self, "_tied_weights_keys", None)\n')
        f.write('    if res is None: return {}\n')
        f.write('    if isinstance(res, list): return {k: None for k in res}\n')
        f.write('    return res\n')
        f.write('def _safe_tied_keys_setter(self, value):\n')
        f.write('    self._tied_weights_keys = value\n')
        f.write('PreTrainedModel.all_tied_weights_keys = property(_safe_tied_keys_getter, _safe_tied_keys_setter)\n')
    print("✅ Đã cấy ghép an toàn vào modeling_utils.py")

print("🎉 Đã vá xong toàn bộ lỗi thư viện trên môi trường Windows!")
