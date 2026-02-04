from paddleocr import PaddleOCR

# Initialize OCR once (CPU safe)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    show_log=False
)

def extract_text(image_path):
    result = ocr.ocr(image_path)
    extracted = []

    for line in result:
        for word in line:
            extracted.append({
                "text": word[1][0],
                "confidence": float(word[1][1]),
                "bbox": word[0]
            })

    return extracted
