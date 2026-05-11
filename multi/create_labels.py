import os
import cv2
import torch
import numpy as np

from crnn.crnn_model import CRNN
from crnn.utils import ALPHABET, indices_to_text

# ========= 設定 =========
INPUT_DIR = "crop"
RESIZED_DIR = "crnn/resized"
OUTPUT_LABEL_TXT = "crnn/test_labels.txt"
CRNN_MODEL = "models/crnn_label/crnn-best.pth"
DEVICE = "cpu"

# CRNN入力サイズ (W, H)
TARGET_SIZE = (100, 32)

# 保存設定
SAVE_PROCESSED_IMAGE = True


# ========= グレースケール画像を pad 付きで 100x32 に整形 =========
def resize_with_pad_gray(img_gray, size=(100, 32)):
    target_w, target_h = size

    if img_gray is None or img_gray.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    h, w = img_gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 黒背景で pad
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ========= 前処理 =========
def preprocess_for_crnn(img_bgr):
    # CRNN(nc=1) なので最後は1chにする必要がある
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 二値化やCLAHEはしない
    pad_img = resize_with_pad_gray(gray, size=TARGET_SIZE)

    return pad_img


# ========= CRNN モデル読み込み =========
def load_crnn_model():
    print("Loading CRNN model...")

    num_classes = len(ALPHABET) + 1
    crnn = CRNN(
        nclass=num_classes,
        nh=256,
        imgH=32,
        nc=1
    )

    state = torch.load(CRNN_MODEL, map_location=DEVICE)
    crnn.load_state_dict(state)
    crnn.to(DEVICE)
    crnn.eval()
    return crnn


# ========= 推論 =========
def predict_text(crnn, img_gray_100x32):
    img_norm = img_gray_100x32.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = crnn(tensor)
        logits = logits.permute(1, 0, 2)
        preds = logits.softmax(2).argmax(2)[0].cpu().numpy().tolist()
        text = indices_to_text(preds)

    return text


# ========= メイン処理 =========
def main():
    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL_TXT), exist_ok=True)

    crnn = load_crnn_model()

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    print(f"処理開始: {INPUT_DIR}")
    print(f"対象画像数: {len(image_files)}")

    if len(image_files) == 0:
        print("対象画像が見つかりませんでした。")
        return

    with open(OUTPUT_LABEL_TXT, "w", encoding="utf-8") as out_f:
        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"画像を読み込めませんでした: {img_path}")
                out_f.write(f"{img_name} \n")
                continue

            # 前処理
            processed = preprocess_for_crnn(img)

            # 前処理後画像を保存
            if SAVE_PROCESSED_IMAGE:
                save_path = os.path.join(RESIZED_DIR, img_name)
                cv2.imwrite(save_path, processed)

            # CRNN推論
            text = predict_text(crnn, processed).strip()

            # label.txt に "ファイル名 ラベル" 形式で書き込み
            out_f.write(f"{img_name} {text}\n")

            print(f"{img_name} -> {text}")

    print("完了しました。")
    print(f"前処理画像保存先: {RESIZED_DIR}")
    print(f"ラベル出力先: {OUTPUT_LABEL_TXT}")


if __name__ == "__main__":
    main()