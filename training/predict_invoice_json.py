import torch
from PIL import Image

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from ocr import tesseract_ocr
from training.ml_field_extractor import merge_tokens_by_label, build_invoice_json


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


def predict_invoice(image_path):
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

    word_ids = encoding.word_ids(batch_index=0)
    encoding_tensors = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding_tensors)

    preds = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()

    # map each OCR word -> label
    word_preds = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        label = model.config.id2label[preds[idx]]
        if wid not in word_preds:
            word_preds[wid] = label
        else:
            if word_preds[wid] == "O" and label != "O":
                word_preds[wid] = label

    predicted_labels = [word_preds.get(i, "O") for i in range(len(words))]

    label_text_map = merge_tokens_by_label(words, predicted_labels)
    invoice_json = build_invoice_json(label_text_map)

    return invoice_json, label_text_map


if __name__ == "__main__":
    img_path = "data/raw/invoice_0001.jpg"
    invoice_json, raw_labels = predict_invoice(img_path)

    print("\n✅ RAW LABEL OUTPUT:")
    print(raw_labels)

    print("\n✅ FINAL INVOICE JSON:")
    print(invoice_json)
