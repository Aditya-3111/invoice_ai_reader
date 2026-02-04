def is_valid_amount(value):
    """
    Strong validation for invoice total amounts
    """
    raw = value.strip()

    # Reject percentages
    if "%" in raw:
        return False

    # Reject alphabetic characters
    if any(c.isalpha() for c in raw):
        return False

    # Remove commas
    cleaned = raw.replace(",", "")

    # Must be numeric
    try:
        num = float(cleaned)
    except:
        return False

    # ❌ Reject phone-number-like values
    # (too many digits, no decimal)
    digits_only = cleaned.replace(".", "")
    if len(digits_only) >= 9 and "." not in cleaned:
        return False

    # ❌ Reject unrealistically large invoice totals
    if num > 10_000_000:   # 1 crore threshold (adjustable)
        return False

    # ❌ Reject very small values
    if num < 10:
        return False

    return True
