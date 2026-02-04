import pytesseract
import cv2

# Force pytesseract to use installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(image_path):
    """
    Extract text + bounding boxes from an image using Tesseract OCR
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")


    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    extracted = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = data["conf"][i]

        if text and conf != "-1":
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            extracted.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
            })

    return extracted