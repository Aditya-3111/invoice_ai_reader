from layer2_field_resolver.value_patterns import PATTERNS


def detect_values(tokens):
    """
    Detect values purely by their format
    """
    results = []

    for token in tokens:
        text = token.text.strip()

        for field, pattern in PATTERNS.items():
            if pattern.search(text):
                results.append({
                    "field": field,
                    "value": text,
                    "confidence": token.confidence,
                    "token": token
                })

    return results
