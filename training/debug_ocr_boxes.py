import os
from PIL import Image, ImageDraw
from ocr import tesseract_ocr

IMG_PATH = "data/raw/invoice_0001.jpg"

OUT_PATH = "training/debug_ocr.png"

ocr_data = tesseract_ocr.extract_text(IMG_PATH)

img = Image.open(IMG_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

for item in ocr_data:
    b = item["bbox"]

    # assume b is 4-point bbox
    x1, y1 = b[0]
    x2, y2 = b[2]

    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

img.save(OUT_PATH)
print("✅ Saved:", OUT_PATH)
