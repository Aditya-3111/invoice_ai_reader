import re

PATTERNS = {
    "invoice_number": re.compile(r"[A-Z0-9]{3,}[/\-][A-Z0-9\-]+", re.I),
    "date": re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4})\b"
    ),
    "amount": re.compile(
        r"(₹|Rs\.?|INR)?\s?\d{1,3}(,\d{3})*(\.\d{2})?"
    ),
    "gstin": re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{3}\b"
    ),
    "pan": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    )
}
