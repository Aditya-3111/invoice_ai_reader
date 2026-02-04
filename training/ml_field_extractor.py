from collections import defaultdict

def merge_tokens_by_label(words, labels):
    """
    words = ["Invoice", "No:", "TXN-123"]
    labels = ["O", "O", "INVOICE_NO"]
    """
    grouped = defaultdict(list)

    for w, lab in zip(words, labels):
        if lab == "O":
            continue
        grouped[lab].append(w)

    # merge
    merged = {}
    for lab, toks in grouped.items():
        merged[lab] = " ".join(toks)

    return merged


def build_invoice_json(label_text_map):
    """
    Convert labels into invoice final response format
    """

    def get(label):
        return label_text_map.get(label)

    result = {
        "invoice_no": get("INVOICE_NO"),
        "invoice_date": get("INVOICE_DATE"),
        "total_amount": get("TOTAL_AMOUNT"),
        "gst_no": get("GST_NO"),
        "pan_no": get("PAN_NO"),
        "buyer_name": get("BUYER_NAME"),
        "buyer_address": get("BUYER_ADDRESS"),
        "buyer_state": get("BUYER_STATE"),
        "seller_name": get("SELLER_NAME"),
        "seller_address": get("SELLER_ADDRESS"),
        "seller_state": get("SELLER_STATE"),
        "buyer_phone": get("BUYER_PHONE"),
        "buyer_email": get("BUYER_EMAIL"),
        "seller_phone": get("SELLER_PHONE"),
        "seller_email": get("SELLER_EMAIL"),
        "bank_name": get("BANK_NAME"),
        "bank_acc_no": get("BANK_ACC_NO"),
        "bank_ifsc": get("BANK_IFSC"),
        "cgst": get("CGST"),
        "sgst": get("SGST"),
        "igst": get("IGST"),
    }

    return result
