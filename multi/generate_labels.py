import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from crnn.crnn_model import CRNN
from crnn.utils import ALPHABET, indices_to_text

VIDEO_PATH = "/mnt/g/共有ドライブ/量子情報デバイス研究室/修士2026/髙橋大騎/machine2の動画/machine2_学習+1.mp4"

YOLO_MODEL = "models/best.pt"
CRNN_MODEL = "models/crnn_label/crnn-best.pth"

OUTPUT_DIR = "crnn/test_resized"
OUTPUT_LABEL_TXT = "crnn/test_labels.txt"

DEVICE = "cpu"

TARGET_FPS = 10
CONF_THRES = 0.1
TARGET_SIZE = (100, 32)

# ========= 前処理パラメータ =========
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 95

# ========= ノイズ除去設定 =========
REMOVE_SMALL_WHITE_NOISE = False
MAX_NOISE_AREA = 50

# True = 追記, False = 上書き
APPEND_LABEL_FILE = False

# ========= 小さい白ノイズ除去 =========
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


# ========= 前処理 =========
clahe = None
if USE_CLAHE:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )

def preprocess_for_crnn(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)

    # 1) Gチャネル抽出
    gray = crop_bgr[:, :, 1]

    # 2) CLAHE
    if clahe is not None:
        gray = clahe.apply(gray)

    # 3) 二値化
    _, binary = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # 4) 小さい白ノイズ除去
    if REMOVE_SMALL_WHITE_NOISE:
        binary = remove_small_white_components(binary, max_area=MAX_NOISE_AREA)
        
    # 5) リサイズ
    resized = cv2.resize(binary, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return resized


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


# ========= CRNN 推論 =========
def predict_text(crnn, img_gray_100x32):
    img_norm = img_gray_100x32.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = crnn(tensor)
        logits = logits.permute(1, 0, 2)
        preds = logits.softmax(2).argmax(2)[0].cpu().numpy().tolist()
        text = indices_to_text(preds)

    return text.strip()


# ========= メイン処理 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL_TXT), exist_ok=True)

    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL)

    crnn = load_crnn_model()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {VIDEO_PATH}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0

    frame_interval = max(1, int(round(orig_fps / TARGET_FPS)))
    actual_fps = orig_fps / frame_interval
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    print(f"処理開始: {VIDEO_PATH}")
    print(f"入力FPS: {orig_fps:.3f}")
    print(f"指定抽出FPS: {TARGET_FPS}")
    print(f"実際の抽出FPS: {actual_fps:.3f}")
    print(f"frame_interval: {frame_interval}")

    mode = "a" if APPEND_LABEL_FILE else "w"

    frame_idx = 0
    saved_count = 0

    with open(OUTPUT_LABEL_TXT, mode, encoding="utf-8") as out_f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # YOLO 推論
                results = yolo(frame, conf=CONF_THRES, verbose=False)

                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    # クラスごとに confidence 最大の1件だけ残す
                    best_by_class = {}

                    for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
                        if cls_id not in best_by_class or conf > best_by_class[cls_id]["conf"]:
                            best_by_class[cls_id] = {
                                "xyxy": xyxy,
                                "conf": conf,
                            }

                    # このフレーム内の結果をいったんためる
                    frame_records = []

                    for cls_id, item in best_by_class.items():
                        x1, y1, x2, y2 = item["xyxy"].astype(int)
                        conf = item["conf"]

                        H, W = frame.shape[:2]
                        x1 = max(0, min(W - 1, x1))
                        x2 = max(0, min(W, x2))
                        y1 = max(0, min(H - 1, y1))
                        y2 = max(0, min(H, y2))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        if crop is None or crop.size == 0:
                            continue

                        processed = preprocess_for_crnn(crop)

                        time_sec = frame_idx / orig_fps
                        cls_name = yolo.names[int(cls_id)]

                        out_name = f"{video_name}_t{time_sec:06.2f}s_{cls_name}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, out_name)

                        # 前処理画像は常に保存
                        cv2.imwrite(out_path, processed)

                        text = predict_text(crnn, processed)

                        frame_records.append({
                            "cls_name": cls_name,
                            "out_name": out_name,
                            "text": text,
                            "conf": conf,
                        })

                    # left -> right の順に並べ替え
                    class_order = {"left": 0, "right": 1}
                    frame_records.sort(key=lambda x: class_order.get(x["cls_name"], 999))

                    # 並べ替え後に labels.txt へ書き込み
                    for rec in frame_records:
                        out_f.write(f"{rec['out_name']} {rec['text']}\n")
                        saved_count += 1
                        print(f"[{saved_count}] {rec['out_name']} -> {rec['text']} (conf={rec['conf']:.3f})")

            frame_idx += 1

    cap.release()

    print("完了しました。")
    print(f"保存枚数: {saved_count}")
    print(f"前処理画像保存先: {OUTPUT_DIR}")
    print(f"ラベル出力先: {OUTPUT_LABEL_TXT}")


if __name__ == "__main__":
    main()