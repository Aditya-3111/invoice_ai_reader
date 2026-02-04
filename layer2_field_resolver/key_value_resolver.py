from layer2_field_resolver.key_value_scorer import score_value_candidate


def find_value_for_key(key_token, tokens, top_k=3):
    """
    Find best value token for a given key token
    """

    candidates = []

    for token in tokens:
        if token == key_token:
            continue

        score = score_value_candidate(key_token, token)
        candidates.append((token, score))

    # Sort by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[:top_k]
