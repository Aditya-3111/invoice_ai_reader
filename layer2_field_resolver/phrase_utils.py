def group_nearby_tokens(tokens, y_threshold=15):
    """
    Group tokens that are on the same horizontal line
    (used to build phrases like 'Tax Invoice', 'Invoice No')
    """

    lines = []

    for token in tokens:
        placed = False
        _, y = token.center()

        for line in lines:
            _, ly = line[0].center()
            if abs(y - ly) <= y_threshold:
                line.append(token)
                placed = True
                break

        if not placed:
            lines.append([token])

    return lines
