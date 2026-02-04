from layer2_field_resolver.value_validators import is_valid_amount

TOTAL_KEYWORDS = [
    "total",
    "grand total",
    "net amount",
    "amount payable",
    "total amount",
    "balance due",
    "total incl",
    "total including",
    "total with tax",
    "invoice total"
]


def resolve_fields(detected_values):
    """
    Resolve best value per field using context + rules
    """

    resolved = {}

    # --- collect valid amount candidates ---
    amount_candidates = [
        v for v in detected_values
        if v["field"] == "amount" and is_valid_amount(v["value"])
    ]

    if not amount_candidates:
        resolved["total_amount"] = {
            "value": None,
            "confidence": 0.0
        }
        return resolved

    # ============================
    # 1️⃣ CONTEXT-AWARE SELECTION
    # ============================
    context_matches = []
    for v in amount_candidates:
        context = v.get("context", "").lower()
        if any(k in context for k in TOTAL_KEYWORDS):
            context_matches.append(v)

    if context_matches:
        best = max(
            context_matches,
            key=lambda x: float(x["value"].replace(",", ""))
        )

        resolved["total_amount"] = {
            "value": best["value"],
            "confidence": 0.95
        }
        return resolved

    # ============================
    # 2️⃣ FALLBACK (YOUR OLD LOGIC)
    # ============================
    def numeric_value(val):
        return float(val.replace(",", ""))

    decimal_amounts = [
        v for v in amount_candidates if "." in v["value"]
    ]

    if decimal_amounts:
        best = max(decimal_amounts, key=lambda x: numeric_value(x["value"]))
    else:
        best = max(amount_candidates, key=lambda x: numeric_value(x["value"]))

    resolved["total_amount"] = {
        "value": best["value"],
        "confidence": 0.85
    }

    return resolved
