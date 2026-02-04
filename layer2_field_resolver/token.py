class Token:
    def __init__(
        self,
        text,
        bbox,
        embedding,
        confidence=1.0,
        page=1
    ):
        self.text = text
        self.bbox = bbox              # [x1, y1, x2, y2]
        self.embedding = embedding    # 768-dim vector
        self.confidence = confidence
        self.page = page

    def center(self):
        """Return center point of token bbox"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def width(self):
        x1, _, x2, _ = self.bbox
        return x2 - x1

    def height(self):
        _, y1, _, y2 = self.bbox
        return y2 - y1

    def __repr__(self):
        return f"Token(text='{self.text}', conf={self.confidence:.2f})"
