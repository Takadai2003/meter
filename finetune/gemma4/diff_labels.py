import os

GT_LABELS_TXT = "finetuning/check_labels.txt"
PRED_LABELS_TXT = "finetuning/output_readings_dark/another_labels.txt"

SHOW_CORRECT = False
SHOW_MISSING_FILES = False
SHOW_WRONG_FILES = True


def load_labels(path):
    labels = {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) < 2:
                print(f"ラベル形式が不正のためスキップ: {path} line={line_num} text={line}")
                continue

            filename = parts[0]
            label = "".join(parts[1:])

            labels[filename] = label

    return labels


def main():
    gt_labels = load_labels(GT_LABELS_TXT)
    pred_labels = load_labels(PRED_LABELS_TXT)

    gt_files = set(gt_labels.keys())
    pred_files = set(pred_labels.keys())

    common_files = sorted(gt_files & pred_files)
    only_gt_files = sorted(gt_files - pred_files)
    only_pred_files = sorted(pred_files - gt_files)

    correct = []
    wrong = []

    for filename in common_files:
        gt = gt_labels[filename]
        pred = pred_labels[filename]

        if pred == gt:
            correct.append((filename, gt, pred))
        else:
            wrong.append((filename, gt, pred))

    total_common = len(common_files)
    correct_count = len(correct)
    wrong_count = len(wrong)
    accuracy = correct_count / total_common if total_common > 0 else 0.0

    print("========== Gemma 4 ラベル評価結果 ==========")
    print(f"正解ラベルファイル        : {GT_LABELS_TXT}")
    print(f"Gemma4ラベルファイル      : {PRED_LABELS_TXT}")
    print("")
    print(f"正解ラベル数              : {len(gt_labels)}")
    print(f"Gemma4ラベル数            : {len(pred_labels)}")
    print(f"両方に存在する画像数      : {total_common}")
    print(f"正解数                    : {correct_count}")
    print(f"不正解数                  : {wrong_count}")
    print(f"正答率                    : {accuracy:.4f}")
    print("")
    print(f"正解側にのみ存在する画像数: {len(only_gt_files)}")
    print(f"Gemma4側にのみ存在する画像数: {len(only_pred_files)}")
    print("===========================================")

    if SHOW_WRONG_FILES:
        print("")
        print("========== 不一致一覧 ==========")

        if len(wrong) == 0:
            print("不一致はありません")
        else:
            print("filename  正解ラベル  Gemma4ラベル")
            for filename, gt, pred in wrong:
                print(f"{filename}  {gt}  {pred}")

    if SHOW_MISSING_FILES:
        print("")
        print("========== 正解側にのみ存在する画像 ==========")

        if len(only_gt_files) == 0:
            print("なし")
        else:
            for filename in only_gt_files:
                print(f"{filename}  {gt_labels[filename]}")

        print("")
        print("========== Gemma4側にのみ存在する画像 ==========")

        if len(only_pred_files) == 0:
            print("なし")
        else:
            for filename in only_pred_files:
                print(f"{filename}  {pred_labels[filename]}")

    if SHOW_CORRECT:
        print("")
        print("========== 正解一覧 ==========")

        if len(correct) == 0:
            print("正解はありません")
        else:
            print("filename  正解ラベル  Gemma4ラベル")
            for filename, gt, pred in correct:
                print(f"{filename}  {gt}  {pred}")


if __name__ == "__main__":
    main()