import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CNN_TRANSFORMER_DIR = os.path.join(PROJECT_ROOT, "cnn_transformer")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

if CNN_TRANSFORMER_DIR not in sys.path:
    sys.path.append(CNN_TRANSFORMER_DIR)

from cnn_transformer import CRNN
from utils import ALPHABET, indices_to_text


VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_学習+1.mp4"
VIDEO_STEM = Path(VIDEO_PATH).stem

YOLO_MODEL = "models/yolo_machine2/text-best.pt"

CRNN_MODELS = {
    "left": "models/transformer/best-model_left.pth",
    "right": "models/transformer/best-model.pth",
}

OUTPUT_LABELS_TXT = os.path.join(SCRIPT_DIR, "latest_labels.txt")
# "w" で上書き、"a" で追記
LABELS_WRITE_MODE = "a"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRES = 0.1
CHAR_CONF_THRES = 0.4
TARGET_FPS = 10

# クラスごとに信頼度しきい値を変えたい場合はここを使う
CLASS_CHAR_CONF_THRES = {
    "left": 0.4,
    "right": 0.4,
}

TARGET_HEIGHT = 32

USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 95

REMOVE_SMALL_WHITE_NOISE = True
MAX_NOISE_AREA = 50

SAVE_RESIZED_IMAGE = True
SAVE_DIR = os.path.join(SCRIPT_DIR, "output_readings")

TARGET_CLASSES = ["left", "right"]

print("Loading YOLO model...")
yolo = YOLO(YOLO_MODEL)


def resolve_path(path):
    if os.path.isabs(path):
        return path

    return os.path.join(PROJECT_ROOT, path)


def load_crnn_model(model_path):
    abs_model_path = resolve_path(model_path)

    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"CNN+Transformerモデルが見つかりません: {abs_model_path}")

    print(f"Loading CNN+Transformer model: {model_path}")

    num_classes = len(ALPHABET) + 1

    crnn = CRNN(
        nclass=num_classes,
        nh=256,
        imgH=32,
        nc=1
    )

    state = torch.load(abs_model_path, map_location=DEVICE)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
        }

    crnn.load_state_dict(state)
    crnn.to(DEVICE)
    crnn.eval()

    return crnn


def load_crnn_models_by_class():
    models = {}

    for cls_name, model_path in CRNN_MODELS.items():
        models[cls_name] = load_crnn_model(model_path)

    return models


crnn_models = load_crnn_models_by_class()


clahe = None
if USE_CLAHE:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )


def remove_small_white_components(binary_img, max_area=80):
    cleaned = binary_img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

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


def preprocess_for_crnn(crop):
    if crop is None or crop.size == 0:
        return None

    if len(crop.shape) == 3:
        g = crop[:, :, 1]
    else:
        g = crop.copy()

    if clahe is not None:
        g = clahe.apply(g)

    _, binary = cv2.threshold(
        g,
        THRESH_VALUE,
        255,
        cv2.THRESH_BINARY
    )

    if REMOVE_SMALL_WHITE_NOISE:
        binary = remove_small_white_components(
            binary,
            max_area=MAX_NOISE_AREA
        )

    resized_img = resize_keep_height(
        binary,
        target_height=TARGET_HEIGHT
    )

    return resized_img


def predict_with_crnn(crnn_model, resized_img):
    img_norm = resized_img.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5

    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    seq_len = torch.tensor(
        [max(1, tensor.size(-1) // 2)],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        logits = crnn_model(tensor, seq_lens=seq_len)

        logits = logits.permute(1, 0, 2)
        probs = logits.softmax(2)

        pred_indices = probs.argmax(2)[0].cpu().numpy().tolist()
        text = indices_to_text(pred_indices).strip()

        max_probs = probs.max(2).values[0].cpu().numpy()

        char_confs = []
        prev_idx = None

        for t, idx in enumerate(pred_indices):
            if idx != 0 and idx != prev_idx:
                char_confs.append(float(max_probs[t]))
            prev_idx = idx

    return text, char_confs


def is_valid_prediction(text, char_confs, conf_thres):
    if text == "":
        return False

    if len(char_confs) == 0:
        return False

    if len(text) != len(char_confs):
        return False

    if any(conf < conf_thres for conf in char_confs):
        return False

    return True


def get_best_detections_by_class(results):
    best_by_class = {}

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for xyxy, cls_id, conf in zip(boxes, classes, confs):
            cls_name = yolo.names[int(cls_id)]

            if TARGET_CLASSES is not None and cls_name not in TARGET_CLASSES:
                continue

            if cls_name not in best_by_class or float(conf) > best_by_class[cls_name]["conf"]:
                best_by_class[cls_name] = {
                    "xyxy": xyxy,
                    "conf": float(conf),
                    "cls_name": cls_name
                }

    return best_by_class


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("動画を開けませんでした:", VIDEO_PATH)
        raise SystemExit

    print(f"動画読み込み成功: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0

    frame_skip = max(1, int(round(orig_fps / TARGET_FPS)))
    actual_fps = orig_fps / frame_skip

    print(f"入力動画FPS : {orig_fps:.2f}")
    print(f"処理FPS     : {actual_fps:.2f}")
    print(f"DEVICE      : {DEVICE}")
    print(f"出力ラベル  : {OUTPUT_LABELS_TXT}")
    print(f"書き込みモード: {LABELS_WRITE_MODE}")
    print("使用モデル:")
    for cls_name, model_path in CRNN_MODELS.items():
        print(f"  {cls_name}: {model_path}")

    if SAVE_RESIZED_IMAGE:
        os.makedirs(SAVE_DIR, exist_ok=True)

    os.makedirs(os.path.dirname(OUTPUT_LABELS_TXT), exist_ok=True)

    frame_index = 0
    detected_count = 0
    saved_label_count = 0
    invalid_count = 0
    skipped_no_model_count = 0

    invalid_files = []
    skipped_no_model_files = []

    with open(OUTPUT_LABELS_TXT, LABELS_WRITE_MODE, encoding="utf-8") as label_file:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if frame_index % frame_skip != 0:
                continue

            time_sec = frame_index / orig_fps

            results = yolo(
                frame,
                conf=CONF_THRES,
                verbose=False
            )

            best_by_class = get_best_detections_by_class(results)

            if len(best_by_class) == 0:
                print(f"[t={time_sec:.2f}s] 検出なし")
                continue

            frame_outputs = {}

            for cls_name, item in best_by_class.items():
                if cls_name not in crnn_models:
                    skipped_no_model_count += 1
                    skipped_no_model_files.append((time_sec, cls_name))
                    print(f"除外: class={cls_name} に対応するモデルがありません")
                    continue

                crnn_model = crnn_models[cls_name]
                char_conf_thres = CLASS_CHAR_CONF_THRES.get(cls_name, CHAR_CONF_THRES)

                x1, y1, x2, y2 = item["xyxy"].astype(int)

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

                resized_img = preprocess_for_crnn(crop)

                if resized_img is None:
                    continue

                text, char_confs = predict_with_crnn(
                    crnn_model,
                    resized_img
                )

                out_name = f"{VIDEO_STEM}_t{time_sec:06.2f}s_{cls_name}.jpg"

                if SAVE_RESIZED_IMAGE:
                    class_save_dir = os.path.join(SAVE_DIR, cls_name)
                    os.makedirs(class_save_dir, exist_ok=True)

                    out_path = os.path.join(class_save_dir, out_name)
                    cv2.imwrite(out_path, resized_img)

                detected_count += 1

                if not is_valid_prediction(text, char_confs, char_conf_thres):
                    invalid_count += 1
                    invalid_files.append((out_name, cls_name, text, char_confs))

                    print(
                        f"除外: {out_name}  "
                        f"class={cls_name}  "
                        f"text={text}  "
                        f"char_confs={['{:.2f}'.format(c) for c in char_confs]}  "
                        f"thres={char_conf_thres}"
                    )
                    continue

                label_file.write(f"{out_name} {text}\n")
                saved_label_count += 1

                frame_outputs[cls_name] = text

                print(
                    f"採用: {out_name}  "
                    f"class={cls_name}  "
                    f"model={CRNN_MODELS[cls_name]}  "
                    f"text={text}"
                )

            left_text = frame_outputs.get("left", "")
            right_text = frame_outputs.get("right", "")

            if left_text != "" and right_text != "":
                print(f"[t={time_sec:.2f}s] {left_text}×10^{right_text}")
            elif left_text != "":
                print(f"[t={time_sec:.2f}s] {left_text}×10^[?]")
            elif right_text != "":
                print(f"[t={time_sec:.2f}s] [?]×10^{right_text}")
            else:
                print(f"[t={time_sec:.2f}s] 有効な読み取りなし")

    cap.release()

    print("")
    print("完了")
    print(f"検出画像数: {detected_count}")
    print(f"保存ラベル数: {saved_label_count}")
    print(f"低信頼などで除外した数: {invalid_count}")
    print(f"対応モデルなしで除外した数: {skipped_no_model_count}")
    print(f"保存先: {OUTPUT_LABELS_TXT}")

    if len(invalid_files) > 0:
        print("")
        print("除外されたファイル")
        for filename, cls_name, text, char_confs in invalid_files:
            conf_text = ",".join(f"{c:.2f}" for c in char_confs)
            print(f"{filename}  class={cls_name}  text={text}  confs=[{conf_text}]")

    if len(skipped_no_model_files) > 0:
        print("")
        print("対応モデルがなく除外された検出")
        for time_sec, cls_name in skipped_no_model_files:
            print(f"t={time_sec:.2f}s  class={cls_name}")


if __name__ == "__main__":
    main()