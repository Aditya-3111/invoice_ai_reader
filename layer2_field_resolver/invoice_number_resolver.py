import re

INVOICE_NO_REGEX = re.compile(
    r"\b[A-Z0-9]{2,}[/\-][A-Z0-9\-]{2,}\b",
    re.I
)


def is_valid_invoice_number(text):
    if len(text) < 6:
        return False

    if text.isdigit():
        return False

    if INVOICE_NO_REGEX.search(text):
        return True

    return False


def resolve_invoice_number(tokens):
    candidates = []

    for t in tokens:
        txt = t.text.strip()
        if is_valid_invoice_number(txt):
            candidates.append({
                "value": txt,
                "confidence": t.confidence,
                "y": t.center()[1]
            })

    if not candidates:
        return None

    # Prefer top-most invoice number
    best = min(candidates, key=lambda x: x["y"])
    return best["value"]
