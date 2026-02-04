import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from ocr import tesseract_ocr

MODEL_DIR = "models/invoice_layoutlmv3"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


def normalize_bbox(bbox, w, h):
    return [
        int(1000 * bbox[0][0] / w),
        int(1000 * bbox[0][1] / h),
        int(1000 * bbox[2][0] / w),
        int(1000 * bbox[2][1] / h),
    ]


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    ocr_data = tesseract_ocr.extract_text(image_path)
    words = [x["text"] for x in ocr_data]
    boxes = [normalize_bbox(x["bbox"], w, h) for x in ocr_data]

    encoding = processor(
        img,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # ✅ keep BatchEncoding for word_ids
    word_ids = encoding.word_ids(batch_index=0)

    # ✅ move only tensors to device
    encoding_tensors = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding_tensors)

    preds = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()


    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        label_id = preds[idx]
        label = model.config.id2label[label_id]

        # choose first non-O label for that word
        if word_id not in word_preds:
            word_preds[word_id] = label
        else:
            if word_preds[word_id] == "O" and label != "O":
                word_preds[word_id] = label

    print("\n✅ WORD-LEVEL PREDICTIONS:\n")
    for i, wtxt in enumerate(words):
        label = word_preds.get(i, "O")
        if label != "O":
            print(f"{wtxt:25s} -> {label}")

    print("\n✅ DONE.")


if __name__ == "__main__":
    predict("data/raw/invoice_0501.jpg")
