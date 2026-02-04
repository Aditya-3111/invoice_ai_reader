import torch
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image

class LayoutLMv3Encoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
        self.model = LayoutLMv3Model.from_pretrained(
            "microsoft/layoutlmv3-base"
        ).to(device)

    def encode(self, image_path, words, boxes):
        """
        image_path: str
        words: List[str]
        boxes: List[List[int]]  # normalized 0-1000
        """

        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        return outputs.last_hidden_state
