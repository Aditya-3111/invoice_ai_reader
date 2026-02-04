from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from PIL import Image

from ocr import tesseract_ocr
from layer1_document_understanding.layoutlm_model import LayoutLMv3Encoder
from layer2_field_resolver.token_builder import build_tokens
from layer2_field_resolver.token_filter import filter_tokens
from layer2_field_resolver.value_detector import detect_values
from layer2_field_resolver.final_resolver import resolve_fields
from layer2_field_resolver.invoice_number_resolver import resolve_invoice_number
from layer2_field_resolver.tax_id_resolver import resolve_gstin_pan

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/extract-invoice", methods=["POST"])
def extract_invoice():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # OCR
    ocr = tesseract_ocr.extract_text(file_path)
    words = [i["text"] for i in ocr]

    with Image.open(file_path) as img:
        w, h = img.size


    boxes = []
    for i in ocr:
        b = i["bbox"]
        boxes.append([
            int(1000 * b[0][0] / w),
            int(1000 * b[0][1] / h),
            int(1000 * b[2][0] / w),
            int(1000 * b[2][1] / h)
        ])

    encoder = LayoutLMv3Encoder()
    embeddings = encoder.encode(file_path, words, boxes)

    tokens = filter_tokens(build_tokens(ocr, embeddings))
    # 🔥 FIX INVALID BBOXES FOR LAYOUTLM
    def sanitize_bbox(b):
        x1, y1, x2, y2 = b
        x1 = max(0, min(999, x1))
        y1 = max(0, min(999, y1))
        x2 = max(x1 + 1, min(1000, x2))
        y2 = max(y1 + 1, min(1000, y2))
        return [x1, y1, x2, y2]

    clean_tokens = []
    for t in tokens:
        b = t["bbox"]
        if b[0] == b[2] or b[1] == b[3]:
            continue  # ❌ drop zero-area tokens
        t["bbox"] = sanitize_bbox(b)
        clean_tokens.append(t)

    tokens = clean_tokens

    
    print("\n========== OCR DEBUG: TOP OF PAGE ==========")
    for t in tokens:
        x, y = t.center()
        if y < 300:  # top 30% of page (invoice header area)
            print(f"TEXT='{t.text}' | CONF={round(t.confidence,2)} | Y={y}")
    print("===========================================\n")

    detected_values = detect_values(tokens)
    amount_data = resolve_fields(detected_values)
    invoice_no = resolve_invoice_number(tokens)
    tax_ids = resolve_gstin_pan(tokens)

    # Cleanup
    try:
        os.remove(file_path)
    except Exception as e:
        print("Cleanup warning:", e)


    return jsonify({
       "invoice_number": {
        "value": invoice_no,
        "confidence": 0.85 if invoice_no else 0.0
    },
    "total_amount": amount_data["total_amount"],
    **tax_ids
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
