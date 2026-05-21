from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os

from crnn.cnn_transformer import CRNN
from crnn.utils import ALPHABET, indices_to_text

VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/WIN_20260416_15_01_28_Pro.mp4"
YOLO_MODEL = "models/yolo_machine2/text-best.pt"
CRNN_MODEL = "crnn/models/hugging_face/Transformer/best_datasets.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRES = 0.1
CHAR_CONF_THRES = 0
TARGET_FPS = 0.2

TARGET_HEIGHT = 32

USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 95

REMOVE_SMALL_WHITE_NOISE = False
MAX_NOISE_AREA = 100

SAVE_RESIZED_IMAGE = True
SAVE_DIR = "cropped"

print("Loading YOLO model...")
yolo = YOLO(YOLO_MODEL)


def load_crnn_model(model_path):
    print(f"Loading CRNN model: {model_path}")

    num_classes = len(ALPHABET) + 1

    crnn = CRNN(
        nclass=num_classes,
        nh=256,
        imgH=32,
        nc=1
    )

    state = torch.load(model_path, map_location=DEVICE)

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


crnn_model = load_crnn_model(CRNN_MODEL)


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

    g = crop[:, :, 1]

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
print(f"使用CRNN    : {CRNN_MODEL}")
print(f"DEVICE      : {DEVICE}")

if SAVE_RESIZED_IMAGE:
    os.makedirs(SAVE_DIR, exist_ok=True)

frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    if frame_index % frame_skip != 0:
        continue

    results = yolo(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

    frame_candidates = {}
    skip_frame = False

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        best_by_class = {}

        for xyxy, cls_id, conf in zip(boxes, classes, confs):
            cls_name = yolo.names[int(cls_id)]

            if cls_name not in ["left", "right"]:
                continue

            if cls_id not in best_by_class or conf > best_by_class[cls_id]["conf"]:
                best_by_class[cls_id] = {
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls_name": cls_name
                }

        for cls_id, item in best_by_class.items():
            cls_name = item["cls_name"]

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

            if not is_valid_prediction(text, char_confs, CHAR_CONF_THRES):
                skip_frame = True
                break

            frame_candidates[cls_name] = {
                "text": text,
                "resized_img": resized_img,
                "char_confs": char_confs
            }

        if skip_frame:
            break

    if skip_frame:
        continue

    time_sec = frame_index / orig_fps

    if SAVE_RESIZED_IMAGE:
        for cls_name, item in frame_candidates.items():
            out_name = f"t_{time_sec:08.2f}s_{cls_name}.jpg"
            out_path = os.path.join(SAVE_DIR, out_name)
            cv2.imwrite(out_path, item["resized_img"])

    left_text = frame_candidates.get("left", {}).get("text", "")
    right_text = frame_candidates.get("right", {}).get("text", "")

    if left_text != "" and right_text != "":
        print(f"[t={time_sec:.2f}s] {left_text}×10^{right_text}")
    elif left_text != "":
        print(f"[t={time_sec:.2f}s] {left_text}×10^[?]")
    elif right_text != "":
        print(f"[t={time_sec:.2f}s] [?]×10^{right_text}")
    else:
        print(f"[t={time_sec:.2f}s] 検出なし")

cap.release()
print("完了")