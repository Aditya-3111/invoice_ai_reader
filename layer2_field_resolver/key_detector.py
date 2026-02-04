from layer2_field_resolver.phrase_utils import group_nearby_tokens


def normalize(text):
    """
    Normalize OCR text to handle common OCR mistakes
    """
    return (
        text.lower()
        .replace("1", "i")
        .replace("l", "i")
        .replace("|", "i")
        .replace("0", "o")
    )


def find_key_phrases(tokens, key_phrases):
    """
    Detect multi-token key phrases like:
    'Tax Invoice', 'Invoice No', 'Bill No'
    """

    detected = []
    lines = group_nearby_tokens(tokens)

    normalized_keys = [normalize(k) for k in key_phrases]

    for line in lines:
        line_text = " ".join(normalize(t.text) for t in line)

        for key in normalized_keys:
            if key in line_text:
                detected.append({
                    "text": key,
                    "tokens": line
                })

    return detected
