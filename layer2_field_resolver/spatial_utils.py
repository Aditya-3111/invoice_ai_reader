import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_right_of(token_a, token_b):
    """
    token_b is right of token_a
    """
    ax, ay = token_a.center()
    bx, by = token_b.center()
    return bx > ax


def is_below(token_a, token_b):
    """
    token_b is below token_a
    """
    ax, ay = token_a.center()
    bx, by = token_b.center()
    return by > ay
