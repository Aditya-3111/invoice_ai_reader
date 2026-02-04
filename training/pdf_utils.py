import os
from pdf2image import convert_from_path

POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

def pdf_to_images(pdf_path, out_dir="training/pdf_pages", dpi=300):
    os.makedirs(out_dir, exist_ok=True)

    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH
    )

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    image_paths = []

    for i, page in enumerate(pages, start=1):
        img_path = os.path.join(out_dir, f"{base}_page_{i}.jpg")
        page.save(img_path, "JPEG")
        image_paths.append(img_path)

    return image_paths
