import cv2
import os
from ultralytics import YOLO

# --- 設定項目 ---
VIDEO_PATH = "/mnt/g/共有ドライブ/量子情報デバイス研究室/修士2026/髙橋大騎/machine2の動画/machine2_shake.mp4"
OUTPUT_DIR = "cropped"
MODEL_PATH = "models/yolo/text-best.pt"

TARGET_FPS = 1
TARGET_SIZE = (100, 32)  # (W, H)

# 出力フォルダ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデルロード
model = YOLO(MODEL_PATH)


def resize_keep_aspect(img, target_size, pad_color=(0, 0, 0)):
    """
    アスペクト比を維持したまま target_size に収める。
    余白は pad_color で埋める。

    img: BGR画像
    target_size: (W, H)
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]

    if w == 0 or h == 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    output = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return output


# 動画読み込み
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"動画を開けませんでした: {VIDEO_PATH}")
    raise SystemExit

# 動画情報
orig_fps = cap.get(cv2.CAP_PROP_FPS)
if not orig_fps or orig_fps <= 0:
    orig_fps = 30.0

frame_interval = max(1, int(round(orig_fps / TARGET_FPS)))
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

print(f"処理開始: {VIDEO_PATH}")
print(f"入力FPS: {orig_fps:.2f} / 抽出FPS: {TARGET_FPS} / interval={frame_interval}")

frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 指定FPSでフレーム抽出
    if frame_idx % frame_interval == 0:
        time_sec = frame_idx / orig_fps

        # YOLO推論
        results = model(frame, verbose=False)

        # 信頼度が最大の検出だけを保持
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
                if conf > best_conf:
                    best_conf = float(conf)
                    best_box = xyxy
                    best_cls_id = int(cls_id)

        # 検出なしの場合はスキップ
        if best_box is not None:
            x1, y1, x2, y2 = best_box

            # 枠外クリップ
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # 無効領域はスキップ
            if x2 > x1 and y2 > y1:
                # 1) crop
                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:
                    # 2) アスペクト比を維持してリサイズ
                    out_img = resize_keep_aspect(crop, TARGET_SIZE, pad_color=(0, 0, 0))

                    if out_img is not None:
                        # クラス名取得
                        cls_name = model.names[best_cls_id]

                        # 保存ファイル名
                        out_name = f"{video_name}_t{time_sec:06.2f}s_{cls_name}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, out_name)

                        cv2.imwrite(out_path, out_img)
                        saved_count += 1

                        print(f"保存: {out_path}  conf={best_conf:.4f}")

        if frame_idx % (frame_interval * 10) == 0:
            print(f"--- 処理中: {frame_idx}フレーム目 (累計保存: {saved_count}枚) ---")

    frame_idx += 1

cap.release()
print(f"完了！ 合計 {saved_count} 枚の画像を保存しました。")