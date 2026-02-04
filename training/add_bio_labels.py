import os
import json

GOLD_DIR = "training/gold"

def add_bio(tokens):
    prev = "O"
    for t in tokens:
        cur = t["label"]

        if cur == "O":
            t["label"] = "O"
        else:
            if prev != cur:
                t["label"] = f"B-{cur}"
            else:
                t["label"] = f"I-{cur}"

        prev = cur
    return tokens

def main():
    count = 0
    for fname in os.listdir(GOLD_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(GOLD_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["tokens"] = add_bio(data["tokens"])

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        count += 1

    print(f"✅ BIO labels added to {count} files")

if __name__ == "__main__":
    main()
