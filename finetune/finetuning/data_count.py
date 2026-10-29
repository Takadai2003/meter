from collections import Counter

LABEL_FILES = "gemma4/gemma4_labels.txt"

def analyze_label_file(path):
    counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            label = parts[1].strip()
            counter[label] += 1

    return counter


def print_result(path, counter):
    print(f"\n=== {path} ===")

    total = sum(counter.values())
    print(f"総サンプル数: {total}")

    def sort_key(x):
        try:
            return float(x[0])
        except:
            return float("inf")

    for label, count in sorted(counter.items(), key=sort_key):
        print(f"{label:>6} : {count}")


def main():
    counter = analyze_label_file(LABEL_FILES)
    print_result(LABEL_FILES, counter)


if __name__ == "__main__":
    main()