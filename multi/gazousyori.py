import cv2
import os
import numpy as np

INPUT_DIR = 'crnn/datasets'
OUTPUT_DIR = 'crnn/resized'

TARGET_SIZE = (100, 32)

# =========================================================
# 各処理の実行 ON/OFF
# =========================================================
RUN_GRAYSCALE = True           # グレースケール化する
RUN_CLAHE = False              # CLAHEをかける
RUN_BINARY = False             # 二値化する
RUN_DENOISE = False            # 小さい白ノイズを除去する
RUN_CROP_FOREGROUND = False    # 白画素領域でトリミングする
RUN_RESIZE_PAD = True          # アスペクト比維持リサイズ + 黒パディングする

# =========================================================
# 各工程の保存 ON/OFF
# =========================================================
SAVE_GRAY = False
SAVE_CLAHE = False
SAVE_BINARY = False
SAVE_DENOISED = False
SAVE_CROPPED = False
SAVE_FINAL = True

# =========================================================
# 各処理のパラメータ
# =========================================================
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

THRESH_VALUE = 85

MAX_NOISE_AREA = 100

CROP_MARGIN = 2

# "center" または "left"
PAD_ALIGN = ""

# 黒背景
PAD_VALUE = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_gray(img):
    """
    入力画像をグレースケールにする。
    すでにグレースケールならそのまま返す。
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def remove_small_white_components(binary_img, max_area=80):
    """
    二値画像から小さい白領域を削除する。
    binary_img は 0/255 の1ch画像を想定。
    """
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


def crop_foreground(img, margin=2):
    """
    白画素、または明るい画素の外接矩形で画像を切り出す。
    img はグレースケールでもカラーでも可。
    """
    if len(img.shape) == 3:
        mask_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mask_src = img

    ys, xs = np.where(mask_src > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    h, w = mask_src.shape[:2]

    x1 = max(xs.min() - margin, 0)
    x2 = min(xs.max() + margin + 1, w)
    y1 = max(ys.min() - margin, 0)
    y2 = min(ys.max() + margin + 1, h)

    return img[y1:y2, x1:x2]


def resize_keep_aspect_with_padding(
    img,
    target_size=(100, 32),
    pad_value=0,
    align="center"
):
    """
    アスペクト比を維持して target_size に収める。
    余った領域は pad_value で埋める。

    target_size は (width, height)。
    グレースケール画像、カラー画像の両方に対応。
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        if len(img.shape) == 3:
            return np.full((target_h, target_w, img.shape[2]), pad_value, dtype=np.uint8)
        return np.full((target_h, target_w), pad_value, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interpolation
    )

    if len(img.shape) == 3:
        canvas = np.full(
            (target_h, target_w, img.shape[2]),
            pad_value,
            dtype=np.uint8
        )
    else:
        canvas = np.full(
            (target_h, target_w),
            pad_value,
            dtype=np.uint8
        )

    if align == "left":
        x_offset = 0
    else:
        x_offset = (target_w - new_w) // 2

    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


clahe = None
if RUN_CLAHE:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )

image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])

print(f"処理開始: {INPUT_DIR}")
print(f"出力先: {OUTPUT_DIR}")
print(f"対象画像数: {len(image_files)}")

print("実行設定:")
print(f"  RUN_GRAYSCALE       = {RUN_GRAYSCALE}")
print(f"  RUN_CLAHE           = {RUN_CLAHE}")
print(f"  RUN_BINARY          = {RUN_BINARY}")
print(f"  RUN_DENOISE         = {RUN_DENOISE}")
print(f"  RUN_CROP_FOREGROUND = {RUN_CROP_FOREGROUND}")
print(f"  RUN_RESIZE_PAD      = {RUN_RESIZE_PAD}")

if len(image_files) == 0:
    print("対象画像が見つかりませんでした。")
    raise SystemExit

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"画像を読み込めませんでした: {img_path}")
        continue

    base_name = os.path.splitext(os.path.basename(img_name))[0]

    current = img.copy()

    # 1) グレースケール変換
    if RUN_GRAYSCALE:
        current = ensure_gray(current)

        if SAVE_GRAY:
            gray_path = os.path.join(OUTPUT_DIR, f"{base_name}_gray.png")
            cv2.imwrite(gray_path, current)
            print(f"保存: {gray_path}")

    # 2) CLAHE
    if RUN_CLAHE:
        gray_for_clahe = ensure_gray(current)
        current = clahe.apply(gray_for_clahe)

        if SAVE_CLAHE:
            clahe_path = os.path.join(OUTPUT_DIR, f"{base_name}_clahe.png")
            cv2.imwrite(clahe_path, current)
            print(f"保存: {clahe_path}")

    # 3) 二値化
    if RUN_BINARY:
        gray_for_binary = ensure_gray(current)

        _, current = cv2.threshold(
            gray_for_binary,
            THRESH_VALUE,
            255,
            cv2.THRESH_BINARY
        )

        if SAVE_BINARY:
            binary_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary.png")
            cv2.imwrite(binary_path, current)
            print(f"保存: {binary_path}")

    # 4) 小さい白ノイズ除去
    if RUN_DENOISE:
        # ノイズ除去は二値画像前提なので、二値化していなければここで二値化する
        if len(current.shape) == 3:
            current_for_denoise = ensure_gray(current)
        else:
            current_for_denoise = current

        unique_values = np.unique(current_for_denoise)
        is_binary_like = np.all(np.isin(unique_values, [0, 255]))

        if not is_binary_like:
            _, current_for_denoise = cv2.threshold(
                current_for_denoise,
                THRESH_VALUE,
                255,
                cv2.THRESH_BINARY
            )

        current = remove_small_white_components(
            current_for_denoise,
            max_area=MAX_NOISE_AREA
        )

        if SAVE_DENOISED:
            cleaned_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary_denoised.png")
            cv2.imwrite(cleaned_path, current)
            print(f"保存: {cleaned_path}")

    # 5) 前景トリミング
    if RUN_CROP_FOREGROUND:
        current = crop_foreground(
            current,
            margin=CROP_MARGIN
        )

        if SAVE_CROPPED:
            cropped_path = os.path.join(OUTPUT_DIR, f"{base_name}_cropped.png")
            cv2.imwrite(cropped_path, current)
            print(f"保存: {cropped_path}")

    # 6) アスペクト比維持リサイズ + 黒パディング
    if RUN_RESIZE_PAD:
        current = resize_keep_aspect_with_padding(
            current,
            target_size=TARGET_SIZE,
            pad_value=PAD_VALUE,
            align=PAD_ALIGN
        )

    # 7) 最終保存
    if SAVE_FINAL:
        # labels.txt のファイル名と一致させるため、元画像名のまま保存
        output_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(output_path, current)

print("完了しました。")