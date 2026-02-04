import os
import torch
from PIL import Image

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from ocr import tesseract_ocr

MODEL_DIR = "models/invoice_layoutlmv3"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def normalize_bbox(bbox, w, h):
    # pixel -> 0..1000
    return [
        int(1000 * bbox[0][0] / w),
        int(1000 * bbox[0][1] / h),
        int(1000 * bbox[2][0] / w),
        int(1000 * bbox[2][1] / h)
    ]


def predict_invoice(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # OCR
    ocr_data = tesseract_ocr.extract_text(image_path)

    words = [x["text"] for x in ocr_data]
    boxes = [normalize_bbox(x["bbox"], w, h) for x in ocr_data]

    enc = processor(
        img,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    logits = out.logits
    preds = logits.argmax(-1)[0].cpu().numpy()

    # token mapping
    input_ids = enc["input_ids"][0].cpu().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    results = []
    for token, pred_id in zip(tokens, preds):
        label = model.config.id2label[int(pred_id)]
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        results.append((token, label))

    return results


if __name__ == "__main__":
    test_img = "data/raw/invoice_0501.jpg"  # change path
    preds = predict_invoice(test_img)

    print("\n✅ PREDICTIONS:\n")
    for t, l in preds:
        if l != "O":
            print(f"{t:25s} -> {l}")
