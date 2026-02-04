import re


def is_valid_text(text):
    """
    Reject garbage OCR tokens
    """
    if len(text) < 2:
        return False

    # Mostly symbols
    if re.fullmatch(r"[\W_]+", text):
        return False

    # Too many repeating characters
    if len(set(text.lower())) <= 2:
        return False

    return True


def filter_tokens(tokens, min_confidence=0.4):
    """
    Remove low-quality tokens
    """
    clean_tokens = []

    for token in tokens:
        if token.confidence < min_confidence:
            continue

        if not is_valid_text(token.text):
            continue

        clean_tokens.append(token)

    return clean_tokens
