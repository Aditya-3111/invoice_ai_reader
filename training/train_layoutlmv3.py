import os
import json
import random
from glob import glob
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    DefaultDataCollator,
    Trainer,
)

# =========================
# PATHS
# =========================
GOLD_DIR = "training/gold"
IMG_DIR = "data/raw"
OUT_MODEL_DIR = "models/invoice_layoutlmv3"

os.makedirs(OUT_MODEL_DIR, exist_ok=True)

# =========================
# ✅ BIO LABELS (MATCH GOLD)
# =========================
LABELS = [
    "O",

    "B-INVOICE_NO", "I-INVOICE_NO",
    "B-INVOICE_DATE", "I-INVOICE_DATE",

    "B-SELLER_NAME", "I-SELLER_NAME",
    "B-SELLER_ADDRESS", "I-SELLER_ADDRESS",
    "B-SELLER_PHONE", "I-SELLER_PHONE",
    "B-SELLER_EMAIL", "I-SELLER_EMAIL",
    "B-SELLER_GST_NO", "I-SELLER_GST_NO",
    "B-SELLER_PAN_NO", "I-SELLER_PAN_NO",
    "B-SELLER_STATE", "I-SELLER_STATE",

    "B-BUYER_NAME", "I-BUYER_NAME",
    "B-BUYER_ADDRESS", "I-BUYER_ADDRESS",
    "B-BUYER_PHONE", "I-BUYER_PHONE",
    "B-BUYER_EMAIL", "I-BUYER_EMAIL",
    "B-BUYER_GST_NO", "I-BUYER_GST_NO",
    "B-BUYER_PAN_NO", "I-BUYER_PAN_NO",
    "B-BUYER_STATE", "I-BUYER_STATE",

    "B-ITEM_NAME", "I-ITEM_NAME",
    "B-ITEM_QTY", "I-ITEM_QTY",
    "B-ITEM_UNIT_RATE", "I-ITEM_UNIT_RATE",

    "B-TOTAL_AMOUNT", "I-TOTAL_AMOUNT",

    "B-BANK_NAME", "I-BANK_NAME",
    "B-ACCOUNT_NUMBER", "I-ACCOUNT_NUMBER",
    "B-IFSC_CODE", "I-IFSC_CODE",

    "B-CGST", "I-CGST",
    "B-SGST", "I-SGST",
    "B-IGST", "I-IGST",
]

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# =========================
# LOAD GOLD DATA
# =========================
def load_gold():
    files = sorted(glob(os.path.join(GOLD_DIR, "*.json")))
    samples = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)

        img_path = os.path.join(IMG_DIR, d["image"])
        if not os.path.exists(img_path):
            continue
        words, boxes, labels = [], [], []

        for t in d["tokens"]:
            txt = t["text"].strip()
            if not txt:
                continue

            words.append(txt)
            boxes.append(t["bbox"])
            labels.append(label2id.get(t["label"], 0))  # default O

        if len(words) < 5:
            continue

        samples.append({
            "image_path": img_path,
            "words": words,
            "boxes": boxes,
            "labels": labels,
        })

    return samples

# =========================
# CUSTOM TRAINER (WEIGHTED LOSS)
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# =========================
# MAIN TRAIN
# =========================
def main():
    samples = load_gold()
    print("✅ Total gold samples:", len(samples))

    if len(samples) < 2:
        raise RuntimeError("❌ Not enough gold data to train")

    # ---- label distribution
    all_labels = []
    for s in samples:
        all_labels.extend(s["labels"])

    counts = Counter(all_labels)
    total = sum(counts.values())

    print("\n✅ Label distribution:")
    for k, v in sorted(counts.items()):
        print(f"   {id2label[k]:25s} -> {v}")

    # ---- class weights
    weights = []
    for i in range(len(LABELS)):
        freq = counts.get(i, 1)
        weights.append(total / freq)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * len(LABELS)

    print("\n✅ Class weights computed\n")

    # ---- train / val split
    random.shuffle(samples)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)

    # ---- processor & model
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    # ---- preprocessing
    def preprocess(ex):
        image = Image.open(ex["image_path"]).convert("RGB")

        enc = processor(
            image,
            ex["words"],
            boxes=ex["boxes"],
            word_labels=ex["labels"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        enc = {k: v.squeeze(0).tolist() for k, v in enc.items()}
        return enc

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    # ---- training args
    args = TrainingArguments(
    output_dir="training/runs/layoutlmv3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    num_train_epochs=10,
    logging_steps=10,
    eval_strategy="epoch",      # ✅ FIXED
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
        class_weights=weights,
    )

    trainer.train()

    model.save_pretrained(OUT_MODEL_DIR)
    processor.save_pretrained(OUT_MODEL_DIR)

    print("\n✅ TRAINING COMPLETE")
    print("✅ Model saved at:", OUT_MODEL_DIR)

# =========================
if __name__ == "__main__":
    main()
