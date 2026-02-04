# 📄 Invoice AI Reader

An end-to-end **AI-powered invoice understanding system** that extracts structured information from invoice documents using **OCR + LayoutLMv3 + rule-based field resolution**.

This project demonstrates a **real-world document AI pipeline** used in enterprise invoice processing systems.

---

## 🚀 Features

- OCR-based token extraction (Tesseract / Paddle OCR)
- Layout-aware token classification using **LayoutLMv3**
- Weak labeling → Gold dataset generation pipeline
- BIO-tag based training for invoice fields
- Intelligent field resolution (regex + spatial + semantic rules)
- JSON output generation for extracted invoice data
- Visualization & debugging tools for OCR and labels

---

## 🧠 Architecture Overview

PDF / Image
↓
OCR (Tokens + Bounding Boxes)
↓
Weak Label Generator
↓
Gold Dataset (BIO Labels)
↓
LayoutLMv3 Training
↓
Token Predictions
↓
Field Resolver (Rules + Context)
↓
Final Structured Invoice JSON


---

## 🗂️ Project Structure

INVOICE_AI_READER/
│
├── app.py
├── requirements.txt
├── README.md
│
├── layer1_document_understanding/
│ ├── layoutlm_model.py
│ ├── cnn_encoder.py
│ ├── rnn_encoder.py
│ └── donut_model.py
│
├── layer2_field_resolver/
│ ├── key_value_resolver.py
│ ├── semantic_matcher.py
│ ├── regex_engine.py
│ ├── spatial_utils.py
│ └── tax_id_resolver.py
│
├── ocr/
│ ├── tesseract_ocr.py
│ └── paddle_ocr.py
│
├── training/
│ ├── weak_label_generator.py
│ ├── convert_labelstudio_export.py
│ ├── train_layoutlmv3.py
│ ├── predict_layoutlmv3.py
│ └── visualize_labels.py
│
├── utils/
│ ├── image_utils.py
│ ├── layout_utils.py
│ ├── pdf_utils.py
│ └── logger.py
│
└── tools/
└── image_server.py


---

## 🏷️ Supported Invoice Fields

- Invoice Number
- Invoice Date
- Seller / Buyer Name
- Seller / Buyer Address
- GST / PAN
- Phone & Email
- Item Name & Unit Rate
- CGST / SGST / IGST
- Total Amount
- Bank Details (Account No, IFSC)

---

## ⚙️ Installation

```bash
git clone https://github.com/<Aditya-3111>/invoice_ai_reader.git
cd invoice_ai_reader
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
🏋️ Model Training
python -m training.weak_label_generator
python -m training.convert_labelstudio_export
python -m training.train_layoutlmv3
🔍 Inference (Prediction)
python -m training.predict_invoice_json --image path/to/invoice.jpg
📦 Output Example
{
  "invoice_no": "2023-001",
  "invoice_date": "22-04-2023",
  "seller_name": "JD Enterprises",
  "total_amount": "18355.00",
  "cgst": "9%",
  "sgst": "9%"
}
🛠️ Tech Stack
Python 3.10

PyTorch

HuggingFace Transformers

LayoutLMv3

Tesseract OCR / Paddle OCR

Label Studio

OpenCV, PIL

🎯 Use Cases
Automated invoice processing

Accounts payable automation

Enterprise document AI systems

OCR + NLP research projects

📌 Note
Model weights, training outputs, and datasets are intentionally excluded from the repository.

👨‍💻 Author
Aditya Shukla
AI / ML | Computer Vision | Document AI

