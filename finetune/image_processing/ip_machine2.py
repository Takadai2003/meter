import cv2
import os
from pathlib import Path

INPUT_DIR = "gemma4/gemma4_images"
LABELS_TXT = "gemma4/gemma4_labels.txt"
OUTPUT_DIR = "image_processing/finetune_images"

TARGET_HEIGHT = 32

USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 95

REMOVE_SMALL_WHITE_NOISE = True
MAX_NOISE_AREA = 50

SAVE_GRAY = False
SAVE_CLAHE = False
SAVE_BINARY = False
SAVE_DENOISED = False
SAVE_RESIZED = True

CLEAR_OUTPUT_DIR = True


def load_label_filenames(labels_txt):
    filenames = []

    if not os.path.exists(labels_txt):
        raise FileNotFoundError(f"ラベルファイルが見つかりません: {labels_txt}")

    with open(labels_txt, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) < 2:
                print(f"形式不正のためスキップ: line={line_num} text={line}")
                continue

            filename = parts[0]
            filenames.append(filename)

    return filenames


def clear_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for path in Path(output_dir).iterdir():
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            path.unlink()


def remove_small_white_components(binary_img, max_area=80):
    cleaned = binary_img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    for label in range(1, num_labels):
        _, _, _, _, area = stats[label]

        if area <= max_area:
            cleaned[labels == label] = 0

    return cleaned


def resize_keep_height(img, target_height=32):
    h, w = img.shape[:2]

    if w <= 0 or h <= 0:
        return None

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(
        img,
        (new_w, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def save_debug_image(save_flag, output_dir, base_name, suffix, img):
    if not save_flag:
        return

    path = os.path.join(output_dir, f"{base_name}_{suffix}.jpg")
    cv2.imwrite(path, img)
    print(f"保存: {path}")


def main():
    if CLEAR_OUTPUT_DIR:
        clear_output_dir(OUTPUT_DIR)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = load_label_filenames(LABELS_TXT)

    print("処理開始")
    print(f"入力画像フォルダ: {INPUT_DIR}")
    print(f"入力ラベル: {LABELS_TXT}")
    print(f"出力画像フォルダ: {OUTPUT_DIR}")
    print(f"ラベル記載画像数: {len(image_files)}")

    if len(image_files) == 0:
        print("ラベルファイルに対象画像がありません。")
        return

    clahe = None
    if USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_TILE
        )

    success_count = 0
    missing_count = 0
    read_error_count = 0
    resize_error_count = 0

    missing_files = []
    read_error_files = []
    resize_error_files = []

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)

        if not os.path.exists(img_path):
            missing_count += 1
            missing_files.append(img_name)
            print(f"画像なし: {img_path}")
            continue

        img = cv2.imread(img_path)

        if img is None:
            read_error_count += 1
            read_error_files.append(img_name)
            print(f"画像を読み込めませんでした: {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_name))[0]

        crop = img.copy()

        if len(crop.shape) == 3:
            gray = crop[:, :, 1]
        else:
            gray = crop.copy()

        save_debug_image(SAVE_GRAY, OUTPUT_DIR, base_name, "gray", gray)

        if clahe is not None:
            clahe_img = clahe.apply(gray)
        else:
            clahe_img = gray.copy()

        save_debug_image(SAVE_CLAHE, OUTPUT_DIR, base_name, "clahe", clahe_img)

        _, binary = cv2.threshold(
            clahe_img,
            THRESH_VALUE,
            255,
            cv2.THRESH_BINARY
        )

        save_debug_image(SAVE_BINARY, OUTPUT_DIR, base_name, "binary", binary)

        if REMOVE_SMALL_WHITE_NOISE:
            binary = remove_small_white_components(
                binary,
                max_area=MAX_NOISE_AREA
            )

        save_debug_image(SAVE_DENOISED, OUTPUT_DIR, base_name, "binary_denoised", binary)

        resized = resize_keep_height(
            binary,
            target_height=TARGET_HEIGHT
        )

        if resized is None:
            resize_error_count += 1
            resize_error_files.append(img_name)
            print(f"リサイズ失敗: {img_name}")
            continue

        output_path = os.path.join(OUTPUT_DIR, img_name)

        if SAVE_RESIZED:
            cv2.imwrite(output_path, resized)

        success_count += 1

        print(
            f"保存: {output_path}  "
            f"size={resized.shape[1]}x{resized.shape[0]}"
        )

    print("")
    print("完了しました。")
    print(f"ラベル記載画像数: {len(image_files)}")
    print(f"変換成功: {success_count}")
    print(f"画像なし: {missing_count}")
    print(f"読み込み失敗: {read_error_count}")
    print(f"リサイズ失敗: {resize_error_count}")

    if len(missing_files) > 0:
        print("")
        print("画像が存在しなかったファイル")
        for name in missing_files:
            print(name)

    if len(read_error_files) > 0:
        print("")
        print("読み込み失敗ファイル")
        for name in read_error_files:
            print(name)

    if len(resize_error_files) > 0:
        print("")
        print("リサイズ失敗ファイル")
        for name in resize_error_files:
            print(name)


if __name__ == "__main__":
    main()