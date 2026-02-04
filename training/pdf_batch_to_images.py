import os
from glob import glob
from training.pdf_utils import pdf_to_images

PDF_DIR = r"data/raw"
OUT_DIR = r"data/raw"

def main():
    pdfs = glob(os.path.join(PDF_DIR, "*.pdf"))
    print("✅ Found PDFs:", len(pdfs))

    for pdf_path in pdfs:
        pdf_to_images(pdf_path, OUT_DIR)

    print("✅ PDF conversion completed")

if __name__ == "__main__":
    main()
