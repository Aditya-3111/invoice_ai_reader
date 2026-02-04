def confidence_from_value(value_present, base=0.6):
    """
    Generic confidence calculator
    """
    if value_present:
        return round(min(base + 0.3, 0.99), 2)
    return 0.0
