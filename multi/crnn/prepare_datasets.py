from datasets import load_dataset
from pathlib import Path

IMG_DIR = Path("crnn/datasets")
LABELS_TXT = Path("crnn/labels_datasets.txt")

IMG_DIR.mkdir(parents=True, exist_ok=True)
LABELS_TXT.parent.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("MiXaiLL76/7SEG_OCR")["train"]

with open(LABELS_TXT, "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        image = row["image"].convert("RGB")
        label = str(row["text"])

        filename = f"hf_{i:04d}.png"
        image_path = IMG_DIR / filename

        image.save(image_path)
        f.write(f"{filename} {label}\n")

print("完了")
print(f"画像保存先: {IMG_DIR}")
print(f"ラベル: {LABELS_TXT}")
print(f"件数: {len(dataset)}")