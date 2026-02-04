import os
import json

DATA_DIR = "training/data"          # OCR + weak label jsons
IMG_DIR = "data/raw"
OUT_DIR = "training/labelstudio"
OUT_FILE = os.path.join(OUT_DIR, "tasks.json")

os.makedirs(OUT_DIR, exist_ok=True)

tasks = []
missing = 0

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
        d = json.load(f)

    img_name = d.get("image")
    tokens = d.get("tokens", [])

    if not img_name or not tokens:
        continue

    abs_img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(abs_img_path):
        missing += 1
        continue

    # ---------- OCR TEXT ----------
    words = [t["text"] for t in tokens]
    ocr_text = " ".join(words)

    # ---------- WEAK LABELS ----------
    predictions = []
    char_idx = 0

    for t in tokens:
        text = t["text"]
        label = t.get("label", "O")

        start = char_idx
        end = start + len(text)

        if label != "O":
            predictions.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [label]
                }
            })

        char_idx = end + 1  # space

    task = {
        "data": {
            "image": f"http://127.0.0.1:5005/images/{img_name}",
            "ocr_text": ocr_text
        },
        "predictions": [
            {
                "model_version": "weak-v1",
                "score": 0.7,
                "result": predictions
            }
        ],
        "meta": {
            "source_json": fname
        }
    }

    tasks.append(task)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)

print(f"✅ Saved tasks: {OUT_FILE}")
print(f"✅ Total tasks: {len(tasks)}")
print(f"⚠️ Missing images skipped: {missing}")
