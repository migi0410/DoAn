from paddleocr import PaddleOCR
import os

print("Initializing PaddleOCR with PP-OCRv4...")
ocr = PaddleOCR(use_angle_cls=False, lang="vi", enable_mkldnn=False, ocr_version="PP-OCRv4")

# Search for any JPG file in mcocr/train_images
images_dir = "mcocr/train_images"
found_img = None
for root, dirs, files in os.walk(images_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.png')):
            found_img = os.path.join(root, f)
            break
    if found_img:
        break

if found_img:
    print(f"Testing OCR on: {found_img}")
    try:
        result = ocr.ocr(found_img)
        print("OCR run completed successfully!")
        if result and result[0]:
            print(f"Detected {len(result[0])} text boxes.")
            print(f"First text: {result[0][0][1][0]} (confidence: {result[0][0][1][1]})")
        else:
            print("No text detected, but no crash occurred.")
    except Exception as e:
        print(f"OCR CRASHED with error: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("No images found to test.")
