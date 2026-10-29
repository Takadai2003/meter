import os

INPUT_LABELS = "gemma4/gemma4_labels.txt"
LEFT_LABELS = "gemma4/gemma4_labels_left.txt"
RIGHT_LABELS = "gemma4/gemma4_labels_right.txt"

left_count = 0
right_count = 0
skip_count = 0

with open(INPUT_LABELS, "r", encoding="utf-8") as f_in, \
     open(LEFT_LABELS, "w", encoding="utf-8") as f_left, \
     open(RIGHT_LABELS, "w", encoding="utf-8") as f_right:

    for line in f_in:
        line = line.strip()

        if line == "":
            continue

        parts = line.split(maxsplit=1)

        if len(parts) < 2:
            skip_count += 1
            continue

        filename = parts[0]

        if filename.endswith("_left.jpg") or filename.endswith("_left.png"):
            f_left.write(line + "\n")
            left_count += 1

        elif filename.endswith("_right.jpg") or filename.endswith("_right.png"):
            f_right.write(line + "\n")
            right_count += 1

        else:
            print(f"クラス判定できないためスキップ: {filename}")
            skip_count += 1

print("完了")
print(f"left : {left_count}")
print(f"right: {right_count}")
print(f"skip : {skip_count}")