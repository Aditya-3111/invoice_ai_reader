def normalize_bbox(bbox, width, height):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]

    return [
        int(1000 * min(x_coords) / width),
        int(1000 * min(y_coords) / height),
        int(1000 * max(x_coords) / width),
        int(1000 * max(y_coords) / height)
    ]
