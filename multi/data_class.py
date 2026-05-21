import os
import shutil

# ===== 実行モード =====
PROCESS_IMAGES = True
PROCESS_LABELS = True

# ===== 入力 =====
IMAGE_DIR = "crnn/test_resized"
LABEL_FILE = "crnn/test_labels.txt"

# ===== 出力 =====
LEFT_DIR = "crnn/resized_left"
RIGHT_DIR = "crnn/resized_right"

LABEL_LEFT = "crnn/labels_left.txt"
LABEL_RIGHT = "crnn/labels_right.txt"

os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

image_exts = (".jpg", ".jpeg", ".png", ".bmp")

# ===== 画像振り分け =====
left_count = 0
right_count = 0
skip_count = 0

if PROCESS_IMAGES:
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith(image_exts):
            continue

        src_path = os.path.join(IMAGE_DIR, fname)

        if "_left" in fname:
            dst_path = os.path.join(LEFT_DIR, fname)
            shutil.move(src_path, dst_path)
            left_count += 1

        elif "_right" in fname:
            dst_path = os.path.join(RIGHT_DIR, fname)
            shutil.move(src_path, dst_path)
            right_count += 1

        else:
            skip_count += 1
            print(f"skip image: {fname}")

# ===== ラベル振り分け =====
label_left_count = 0
label_right_count = 0

if PROCESS_LABELS:
    with open(LABEL_FILE, "r", encoding="utf-8") as infile, \
         open(LABEL_LEFT, "a", encoding="utf-8") as left_out, \
         open(LABEL_RIGHT, "a", encoding="utf-8") as right_out:

        for line in infile:
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split(" ", 1)

            filename = parts[0]

            label = parts[1] if len(parts) > 1 else ""

            if "_left" in filename:
                left_out.write(f"{filename} {label}\n")
                label_left_count += 1

            elif "_right" in filename:
                right_out.write(f"{filename} {label}\n")
                label_right_count += 1

# ===== 結果表示 =====
print("=== 完了 ===")

if PROCESS_IMAGES:
    print(f"画像 left : {left_count} 枚")
    print(f"画像 right: {right_count} 枚")
    print(f"画像 skip : {skip_count} 枚")
else:
    print("画像処理は実行していません。")

print()

if PROCESS_LABELS:
    print(f"ラベル left : {label_left_count} 行")
    print(f"ラベル right: {label_right_count} 行")
else:
    print("ラベル処理は実行していません。")