import re

GST_REGEX = re.compile(
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b"
)

PAN_REGEX = re.compile(
    r"\b[A-Z]{5}\d{4}[A-Z]\b"
)


def resolve_gstin_pan(tokens):
    result = {}

    for t in tokens:
        text = t.text.strip().upper()

        if "gstin" not in result and GST_REGEX.fullmatch(text):
            result["gstin"] = text

        if "pan" not in result and PAN_REGEX.fullmatch(text):
            result["pan"] = text

    return result
