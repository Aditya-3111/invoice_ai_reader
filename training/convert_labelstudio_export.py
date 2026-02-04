import os
import json
from PIL import Image

# ========= PATHS =========
EXPORT_FILE = "training/labelstudio_exports/project-5-at-2026-01-27-15-47-e3591abb.json"
DATA_DIR = "training/data"          # weak OCR jsons (pixel bboxes)
IMG_DIR = "data/raw"                # original images
OUT_DIR = "training/gold"

os.makedirs(OUT_DIR, exist_ok=True)

# ========= LABEL NORMALIZATION =========
LABEL_MAP = {
    "INVOICE_NO": "INVOICE_NO",
    "INVOICE_DATE": "INVOICE_DATE",

    "SELLER_NAME": "SELLER_NAME",
    "SELLER_ADDRESS": "SELLER_ADDRESS",
    "SELLER_PHONE": "SELLER_PHONE",
    "SELLER_EMAIL": "SELLER_EMAIL",
    "SELLER_GST_NO": "SELLER_GST_NO",
    "SELLER_PAN_NO": "SELLER_PAN_NO",
    "SELLER_STATE": "SELLER_STATE",

    "BUYER_NAME": "BUYER_NAME",
    "BUYER_ADDRESS": "BUYER_ADDRESS",
    "BUYER_PHONE": "BUYER_PHONE",
    "BUYER_EMAIL": "BUYER_EMAIL",
    "BUYER_GST_NO": "BUYER_GST_NO",
    "BUYER_PAN_NO": "BUYER_PAN_NO",
    "BUYER_STATE": "BUYER_STATE",

    "ITEM_NAME": "ITEM_NAME",
    "ITEM_QTY": "ITEM_QTY",
    "ITEM_UNIT_RATE": "ITEM_UNIT_RATE",

    "TOTAL_AMOUNT": "TOTAL_AMOUNT",

    "BANK_NAME": "BANK_NAME",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "IFSC_CODE": "IFSC_CODE",

    # lowercase fixes
    "cgst": "CGST",
    "sgst": "SGST",
    "igst": "IGST",
}

# ========= HELPERS =========
def clamp(v):
    return max(0, min(1000, int(v)))

def clamp_bbox(b):
    return [
        clamp(b[0]),
        clamp(b[1]),
        clamp(b[2]),
        clamp(b[3]),
    ]

def inside(rect, tok):
    cx = (tok[0] + tok[2]) / 2
    cy = (tok[1] + tok[3]) / 2
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]

# ========= CORE =========
def convert_one(task):
    source_json = task.get("meta", {}).get("source_json")
    if not source_json:
        return None

    weak_path = os.path.join(DATA_DIR, source_json)
    if not os.path.exists(weak_path):
        return None

    with open(weak_path, "r", encoding="utf-8") as f:
        weak = json.load(f)

    tokens = weak["tokens"]

    # 🔥 CLAMP TOKEN BBOXES + RESET LABELS
    for t in tokens:
        t["bbox"] = clamp_bbox(t["bbox"])
        t["label"] = "O"

    annotations = task.get("annotations", [])
    if not annotations:
        return None

    results = annotations[0].get("result", [])
    if not results:
        return None

    image_path = os.path.join(IMG_DIR, weak["image"])
    if not os.path.exists(image_path):
        return None

    with Image.open(image_path) as img:
        IMG_W, IMG_H = img.size

    # Label Studio rectangles → PIXEL → 0–1000
    for r in results:
        if r.get("type") != "rectanglelabels":
            continue

        raw_label = r["value"]["rectanglelabels"][0]
        label = LABEL_MAP.get(raw_label)
        if not label:
            continue

        v = r["value"]

        px1 = (v["x"] / 100.0) * IMG_W
        py1 = (v["y"] / 100.0) * IMG_H
        px2 = ((v["x"] + v["width"]) / 100.0) * IMG_W
        py2 = ((v["y"] + v["height"]) / 100.0) * IMG_H

        # 🔥 SCALE TO LAYOUTLM SPACE
        rect = clamp_bbox([
            px1 / IMG_W * 1000,
            py1 / IMG_H * 1000,
            px2 / IMG_W * 1000,
            py2 / IMG_H * 1000,
        ])

        for t in tokens:
            if inside(rect, t["bbox"]):
                t["label"] = label

    return {
        "image": weak["image"],
        "tokens": tokens
    }

def main():
    with open(EXPORT_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    count = 0
    for task in tasks:
        converted = convert_one(task)
        if not converted:
            continue

        out_path = os.path.join(OUT_DIR, f"gold_{count:04d}.json")
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(converted, wf, indent=2)

        count += 1

    print(f"✅ Converted {count} tasks into gold dataset at {OUT_DIR}")

if __name__ == "__main__":
    main()
