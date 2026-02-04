import os
import json
from PIL import Image

from ocr import tesseract_ocr
from layer1_document_understanding.layoutlm_model import LayoutLMv3Encoder
from layer2_field_resolver.token_builder import build_tokens
from layer2_field_resolver.token_filter import filter_tokens
from training.pdf_utils import pdf_to_images

OUTPUT_DIR = "training/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_weak_labels(image_path, output_name):
    # 1️⃣ OCR
    ocr_data = tesseract_ocr.extract_text(image_path)

    words = [x["text"] for x in ocr_data]

    # 2️⃣ image size
    with Image.open(image_path) as img:
        w, h = img.size

    boxes = []
    for x in ocr_data:
        (x1, y1), (_, _), (x2, y2), (_, _) = x["bbox"]
        boxes.append([
            int(1000 * x1 / w),
            int(1000 * y1 / h),
            int(1000 * x2 / w),
            int(1000 * y2 / h),
        ])

    # 3️⃣ LayoutLM embeddings
    encoder = LayoutLMv3Encoder()
    embeddings = encoder.encode(image_path, words, boxes)

    # 4️⃣ Token objects
    token_objs = filter_tokens(build_tokens(ocr_data, embeddings))

    # 🔥 CONVERT Token → dict (THIS WAS THE BUG)
    tokens = []
    for t in token_objs:
        tokens.append({
            "text": t.text,
            "bbox": t.bbox,
            "label": "O"
        })

    # 5️⃣ save
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": os.path.basename(image_path),
                "tokens": tokens
            },
            f,
            indent=2
        )

    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    INPUT_DIR = "data/raw"
    files = sorted(os.listdir(INPUT_DIR))

    print(f"📄 Found {len(files)} files in {INPUT_DIR}")

    idx = 1
    for filename in files:
        path = os.path.join(INPUT_DIR, filename)

        try:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                generate_weak_labels(path, f"invoice_{idx:04d}.json")
                idx += 1

            elif filename.lower().endswith(".pdf"):
                for img in pdf_to_images(path):
                    generate_weak_labels(img, f"invoice_{idx:04d}.json")
                    idx += 1

        except Exception as e:
            print(f"❌ Failed {filename} -> {e}")

    print(f"✅ Done. Generated {idx-1} files.")
