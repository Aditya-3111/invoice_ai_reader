import os, json
from PIL import Image

GOLD_DIR = "training/gold"
IMG_DIR = r"C:\labelstudio_files\raw"

def norm_bbox(b, w, h):
    return [
        int(1000 * b[0] / w),
        int(1000 * b[1] / h),
        int(1000 * b[2] / w),
        int(1000 * b[3] / h),
    ]

for fname in os.listdir(GOLD_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(GOLD_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    img_path = os.path.join(IMG_DIR, d["image"])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    w, h = img.size

    for t in d["tokens"]:
        t["bbox"] = norm_bbox(t["bbox"], w, h)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

print("✅ All gold bboxes normalized to 0..1000")
