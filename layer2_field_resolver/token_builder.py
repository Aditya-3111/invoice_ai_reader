import torch
from layer2_field_resolver.token import Token


def build_tokens(ocr_data, embeddings):
    """
    ocr_data: list of dicts from OCR
    embeddings: LayoutLM output [1, T, 768]
    """

    tokens = []
    embedding_vectors = embeddings.squeeze(0)

    for i, item in enumerate(ocr_data):
        if i >= embedding_vectors.size(0):
            break

        token = Token(
            text=item["text"],
            bbox=[
                item["bbox"][0][0],
                item["bbox"][0][1],
                item["bbox"][2][0],
                item["bbox"][2][1],
            ],
            embedding=embedding_vectors[i],
            confidence=item.get("confidence", 1.0),
            page=1
        )

        tokens.append(token)

    return tokens
