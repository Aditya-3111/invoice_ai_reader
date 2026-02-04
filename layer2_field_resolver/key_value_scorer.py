from layer2_field_resolver.spatial_utils import (
    euclidean_distance,
    is_right_of,
    is_below
)


def score_value_candidate(key_token, value_token):
    """
    Higher score = better match
    """

    score = 0.0

    # Position-based scoring
    if is_right_of(key_token, value_token):
        score += 1.0

    if is_below(key_token, value_token):
        score += 0.5

    # Distance penalty
    dist = euclidean_distance(
        key_token.center(),
        value_token.center()
    )

    score -= dist / 500  # normalize

    # OCR confidence
    score += value_token.confidence / 100

    return score
