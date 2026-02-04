import os
import json
import random
from PIL import Image, ImageDraw

DATA_DIR = "training/data"
IMG_DIR = "data/raw"
OUT_DIR = "training/vis"

os.makedirs(OUT_DIR, exist_ok=True)

# basic label colors
LABEL_COLORS = {
    "INVOICE_NO": "blue",
    "INVOICE_DATE": "purple",
    "TOTAL_AMOUNT": "green",
    "GST_NO": "orange",
    "PAN_NO": "orange",
    "BUYER_PHONE": "red",
    "BUYER_EMAIL": "red",
}

def visualize_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data["image"]
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"❌ Image missing: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    label_count = {}

    for tok in data["tokens"]:
        label = tok["label"]
        if label == "O":
            continue

        bbox = tok["bbox"]  # ✅ bbox already in pixel coords
        x1, y1, x2, y2 = bbox

        color = LABEL_COLORS.get(label, "yellow")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), label, fill=color)

        label_count[label] = label_count.get(label, 0) + 1

    out_path = os.path.join(OUT_DIR, os.path.basename(json_path).replace(".json", ".png"))
    img.save(out_path)

    print(f"✅ Saved VIS: {out_path} | labels: {label_count}")


if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    random.shuffle(files)

    sample = files[:20]  # visualize 20 invoices

    for f in sample:
        visualize_one(os.path.join(DATA_DIR, f))

    print("✅ Visualization completed!")
