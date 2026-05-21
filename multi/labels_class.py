import os
import re

LABEL_ALL = "crnn/labels.txt"
LABEL_LEFT = "crnn/labels_left.txt"
LABEL_RIGHT = "crnn/labels_right.txt"


def read_label_file(path):
    records = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            filename, label = parts[0], parts[1].strip()
            records[filename] = label

    return records


def extract_time_sec(filename):
    m = re.search(r"_t(\d+(?:\.\d+)?)s_", filename)
    if m:
        return float(m.group(1))
    return float("inf")


def sort_records(records_dict):
    items = list(records_dict.items())
    items.sort(key=lambda x: extract_time_sec(x[0]))
    return items


def write_label_file(path, records_dict):
    sorted_items = sort_records(records_dict)
    with open(path, "w", encoding="utf-8") as f:
        for filename, label in sorted_items:
            f.write(f"{filename} {label}\n")


def main():
    # 既存の完成済み left/right ラベルを読む
    left_records = read_label_file(LABEL_LEFT)
    right_records = read_label_file(LABEL_RIGHT)

    left_before = len(left_records)
    right_before = len(right_records)

    # 新しい labels.txt を読む
    all_records = read_label_file(LABEL_ALL)

    add_left = 0
    add_right = 0
    skip_other = 0

    for filename, label in all_records.items():
        if "_left" in filename:
            if filename not in left_records:
                left_records[filename] = label
                add_left += 1

        elif "_right" in filename:
            if filename not in right_records:
                right_records[filename] = label
                add_right += 1

        else:
            skip_other += 1

    # 時刻順に並べて保存し直す
    write_label_file(LABEL_LEFT, left_records)
    write_label_file(LABEL_RIGHT, right_records)

    print("完了しました。")
    print(f"left : 既存 {left_before} 行 -> 追加 {add_left} 行 -> 合計 {len(left_records)} 行")
    print(f"right: 既存 {right_before} 行 -> 追加 {add_right} 行 -> 合計 {len(right_records)} 行")
    print(f"left/right 以外でスキップ: {skip_other} 行")


if __name__ == "__main__":
    main()