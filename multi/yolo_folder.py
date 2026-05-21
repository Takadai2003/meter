import cv2
import os
import numpy as np
from ultralytics import YOLO

INPUT_DIR = "crnn/datasets"
OUTPUT_DIR = "crnn/resized"
MODEL_PATH = "models/yolo_hf/text-best.pt"

TARGET_HEIGHT = 32

V_TOP_PERCENT = 27
S_MIN = 45
HUE_TOL = 3
MIN_CANDIDATES = 20
WHITE_S_MAX = 100

OVERWRITE_EXISTING = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def hue_distance(h, target_h):
    diff = np.abs(h.astype(np.int16) - int(target_h))
    return np.minimum(diff, 180 - diff)


def binarize_auto_hue_from_sv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    max_v = int(np.percentile(v, 97.5))
    v_threshold = max_v * (100 - V_TOP_PERCENT) / 100.0

    candidate_mask = (
        (s >= S_MIN) &
        (v >= v_threshold)
    )

    candidate_count = int(np.count_nonzero(candidate_mask))

    if candidate_count < MIN_CANDIDATES:
        white_mask = (
            (s <= WHITE_S_MAX) &
            (v >= v_threshold)
        )

        white_count = int(np.count_nonzero(white_mask))

        if white_count >= MIN_CANDIDATES:
            binary = np.where(white_mask, 255, 0).astype(np.uint8)
            return binary, None, v_threshold, white_count, "white_fallback"

        binary = np.where(v >= v_threshold, 255, 0).astype(np.uint8)
        return binary, None, v_threshold, candidate_count, "v_fallback"

    hist = np.bincount(
        h[candidate_mask].ravel(),
        minlength=180
    )

    target_hue = int(np.argmax(hist))

    hue_mask = hue_distance(h, target_hue) <= HUE_TOL

    final_mask = (
        candidate_mask &
        hue_mask
    )

    binary = np.where(final_mask, 255, 0).astype(np.uint8)

    return binary, target_hue, v_threshold, candidate_count, "auto_hue"


def resize_binary_keep_height(binary_img, target_height=32):
    h, w = binary_img.shape[:2]

    if w <= 0 or h <= 0:
        return None

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(
        binary_img,
        (new_w, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def save_grayscale_image(path, img):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(path, img)


image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
])

print(f"処理開始: {INPUT_DIR}")
print(f"対象画像数: {len(image_files)}")
print(f"V_TOP_PERCENT={V_TOP_PERCENT}, S_MIN={S_MIN}, HUE_TOL={HUE_TOL}")

saved_count = 0
skipped_count = 0
no_detection_count = 0

for img_idx, img_name in enumerate(image_files):
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"読み込めません: {img_name}")
        skipped_count += 1
        continue

    out_name = img_name
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.exists(out_path) and not OVERWRITE_EXISTING:
        print(f"既存ファイルのためスキップ: {out_path}")
        skipped_count += 1
        continue

    results = model(img, verbose=False)

    best_conf = -1.0
    best_box = None
    best_cls_id = None

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
            if float(conf) > best_conf:
                best_conf = float(conf)
                best_box = xyxy
                best_cls_id = int(cls_id)

    if best_box is None:
        print(f"検出なし: {img_name}")
        no_detection_count += 1
        continue

    x1, y1, x2, y2 = best_box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        print(f"無効な検出領域: {img_name}")
        skipped_count += 1
        continue

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"crop失敗: {img_name}")
        skipped_count += 1
        continue

    binary, target_hue, v_threshold, candidate_count, mode = binarize_auto_hue_from_sv(crop)

    resized_binary = resize_binary_keep_height(
        binary,
        target_height=TARGET_HEIGHT
    )

    if resized_binary is None:
        print(f"リサイズ失敗: {img_name}")
        skipped_count += 1
        continue

    save_grayscale_image(out_path, resized_binary)
    saved_count += 1

    cls_name = model.names[best_cls_id]
    hue_text = "None" if target_hue is None else str(target_hue)

    print(
        f"保存: {out_path}  "
        f"size={resized_binary.shape[1]}x{resized_binary.shape[0]}  "
        f"class={cls_name}  "
        f"conf={best_conf:.4f}  "
        f"mode={mode}  "
        f"H={hue_text}  "
        f"Vth={v_threshold:.1f}  "
        f"candidates={candidate_count}"
    )

print("完了！")
print(f"保存: {saved_count} 枚")
print(f"検出なし: {no_detection_count} 枚")
print(f"スキップ: {skipped_count} 枚")