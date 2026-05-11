import cv2
import os

INPUT_DIR = 'crop'
OUTPUT_DIR = 'crnn/test_resized'

TARGET_SIZE = (100, 32)

USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 95
# ノイズ除去設定
REMOVE_SMALL_WHITE_NOISE = True
MAX_NOISE_AREA = 100

# --- 画像保存のON/OFF設定 ---
SAVE_GRAY     = False  # Gチャネル
SAVE_CLAHE    = False  # CLAHE
SAVE_BINARY   = False  # 二値化
SAVE_DENOISED = False   # ノイズ除去
SAVE_RESIZED  = True   # リサイズ

os.makedirs(OUTPUT_DIR, exist_ok=True)

def remove_small_white_components(binary_img, max_area=80):
    cleaned = binary_img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    for label in range(1, num_labels):
        _, _, _, _, area = stats[label]

        if area <= max_area:
            cleaned[labels == label] = 0

    return cleaned


# CLAHE は毎回作らず最初に作成
clahe = None
if USE_CLAHE:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )

# 対象画像一覧
image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])

print(f"処理開始: {INPUT_DIR}")
print(f"対象画像数: {len(image_files)}")

if len(image_files) == 0:
    print("対象画像が見つかりませんでした。")
    raise SystemExit

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"画像を読み込めませんでした: {img_path}")
        continuecreate

    base_name = os.path.splitext(os.path.basename(img_name))[0]

    # 入力画像全体を crop として扱う
    crop = img.copy()

    # 1) Gチャネルのみ使用
    gray = crop[:, :, 1]
    
    if SAVE_GRAY:
        gray_path = os.path.join(OUTPUT_DIR, f"{base_name}_gray.jpg")
        cv2.imwrite(gray_path, gray)
        print(f"保存: {gray_path}")

    # 2) CLAHE
    if clahe is not None:
        clahe_img = clahe.apply(gray)
    else:
        clahe_img = gray.copy()

    if SAVE_CLAHE:
        clahe_path = os.path.join(OUTPUT_DIR, f"{base_name}_clahe.jpg")
        cv2.imwrite(clahe_path, clahe_img)
        print(f"保存: {clahe_path}")

    # 3) 二値化
    _, binary = cv2.threshold(clahe_img, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    if SAVE_BINARY:
        binary_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary.jpg")
        cv2.imwrite(binary_path, binary)
        print(f"保存: {binary_path}")

    # 4) 小さい白ノイズ除去
    if REMOVE_SMALL_WHITE_NOISE:
        binary = remove_small_white_components(binary, max_area=MAX_NOISE_AREA)
        if SAVE_DENOISED:
            cleaned_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary_denoised.jpg")
            cv2.imwrite(cleaned_path, binary)
            print(f"保存: {cleaned_path}")

    # 5) リサイズ
    resized = cv2.resize(binary, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    if SAVE_RESIZED:
        resized_path = os.path.join(OUTPUT_DIR, f"{base_name}.jpg")
        cv2.imwrite(resized_path, resized)
        print(f"保存: {resized_path}")

print("完了しました。")