from training.patterns import (
    GST_REGEX, PAN_REGEX, IFSC_REGEX, EMAIL_REGEX, PHONE_REGEX, DATE_REGEX
)

# Use your existing production resolvers (Frozen logic)
from layer2_field_resolver.invoice_number_resolver import resolve_invoice_number
from layer2_field_resolver.final_resolver import resolve_fields
from layer2_field_resolver.tax_id_resolver import resolve_gstin_pan


def _find_by_regex(tokens, regex):
    found = []
    for t in tokens:
        txt = t.text.strip()
        if regex.fullmatch(txt.upper()):
            found.append(txt)
    return found


def extract_field_values(tokens):
    """
    Returns a dictionary:
    {
      "INVOICE_NO": ["US-001"],
      "TOTAL_AMOUNT": ["20,500.90"],
      "GST_NO": ["09ABCDE..."],
      ...
    }
    """

    extracted = {}

    # -------------------------------
    # 1) Invoice Number
    # -------------------------------
    inv_no = resolve_invoice_number(tokens)
    extracted["INVOICE_NO"] = [inv_no] if inv_no else []

    # -------------------------------
    # 2) Amount (Grand Total)
    # -------------------------------
    # convert tokens -> detected values (amount candidates)
    detected_values = [{"field": "amount", "value": t.text} for t in tokens]
    amount_data = resolve_fields(detected_values)
    total_amount = amount_data.get("total_amount", {}).get("value")
    extracted["TOTAL_AMOUNT"] = [total_amount] if total_amount else []

    # -------------------------------
    # 3) GST & PAN
    # -------------------------------
    tax = resolve_gstin_pan(tokens)
    gst = tax.get("gstin")
    pan = tax.get("pan")

    extracted["GST_NO"] = [gst] if gst else []
    extracted["PAN_NO"] = [pan] if pan else []

    # -------------------------------
    # 4) Emails / Phones
    # -------------------------------
    extracted["BUYER_EMAIL"] = _find_by_regex(tokens, EMAIL_REGEX)
    extracted["SELLER_EMAIL"] = []  # we will split buyer/seller in Phase 2

    extracted["BUYER_PHONE"] = _find_by_regex(tokens, PHONE_REGEX)
    extracted["SELLER_PHONE"] = []  # Phase 2

    # -------------------------------
    # 5) IFSC
    # -------------------------------
    extracted["BANK_IFSC"] = _find_by_regex(tokens, IFSC_REGEX)

    # -------------------------------
    # 6) Invoice Date (weak)
    # -------------------------------
    # Phase 1: just collect all date-like strings.
    extracted["INVOICE_DATE"] = _find_by_regex(tokens, DATE_REGEX)

    # -------------------------------
    # 7) Remaining fields placeholders
    # -------------------------------
    # These will be extracted properly in Phase 2+,
    # for now weak labels will be sparse or empty.

    extracted["BUYER_NAME"] = []
    extracted["BUYER_ADDRESS"] = []
    extracted["BUYER_STATE"] = []

    extracted["SELLER_NAME"] = []
    extracted["SELLER_ADDRESS"] = []
    extracted["SELLER_STATE"] = []

    extracted["BANK_NAME"] = []
    extracted["BANK_ACC_NO"] = []

    extracted["CGST"] = []
    extracted["SGST"] = []
    extracted["IGST"] = []

    extracted["ITEM_NAME"] = []
    extracted["ITEM_DESC"] = []
    extracted["QTY"] = []
    extracted["UNIT_RATE"] = []

    extracted["WARRANTY_DETAILS"] = []
    extracted["WARRANTY_END"] = []
    extracted["WARRANTY_PHONE"] = []

    return extracted
