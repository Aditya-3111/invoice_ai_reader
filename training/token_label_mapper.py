def normalize(text):
    if text is None:
        return ""
    return str(text).strip().lower().replace(",", "")


def assign_labels(tokens, extracted_fields):
    """
    tokens: list of token dicts with keys: text, bbox
    extracted_fields: dict -> field_name : list of values
    """

    labeled_tokens = []

    normalized_field_values = {}
    for field, values in extracted_fields.items():
        normalized_field_values[field] = set(
            normalize(v) for v in values if v
        )

    for t in tokens:
        label = "O"
        t_text = normalize(t["text"])

        for field, values in normalized_field_values.items():
            if t_text in values:
                label = field
                break

        labeled_tokens.append({
            "text": t["text"],
            "bbox": t["bbox"],
            "label": label
        })

    return labeled_tokens
