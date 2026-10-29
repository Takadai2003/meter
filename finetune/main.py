import os
import sys

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cnn_transformer_color.cnn_transformer import CRNN
from cnn_transformer_color.utils import ALPHABET, indices_to_text
from gemma4 import gemma4_labels
from finetuning import finetune_color


# ============================================================
# 共通設定
#
# gemma4_labels.py / finetune_color.py の設定のうち、
# 工程間でそろえる必要があるものをここで上書きする。
# 抽出FPSや学習率などは各スクリプト側の値を使う。
# ============================================================

# 自動ラベル付けに使う動画
VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_slip_train.mp4"

# 表示領域（left / right）を検出するYOLO
YOLO_MODEL_PATH = "models/yolo_machine2/slip.pt"

# Gemma 4が切り出し画像を保存し、ファインチューニングで読み込むフォルダとラベル
IMG_DIR = "finetuning/output_readings_slip/train"
LABELS_TXT = "gemma4/gemma4_labels_slip.txt"

# カラー版CNN+Transformer
BASE_MODEL_PATH = "models/transformer_color/best-model_color_4.pth"
FINETUNED_SAVE_DIR = "models/finetuning_slip"
LATEST_MODEL_PATH = os.path.join(FINETUNED_SAVE_DIR, "latest.pth")

TARGET_CLASSES = ["left", "right"]

CLASS_VALUE_RANGES = {
    "left": (1.0, 9.9),
    "right": (-5, 5),
}

CLASS_EXPECTED_LABEL_STEPS = {
    "left": 0.1,
    "right": 1,
}

# 値1種類あたりのラベル枚数
# MIN: 全ての値がこれに届くまでファインチューニングしない
# MAX: これに達した値はそれ以上集めない
LABEL_MIN_COUNT = 8
LABEL_MAX_COUNT = 10

RUN_GEMMA_LABELS = True
RUN_FINETUNING = True
RUN_LATEST_MODEL_READING = True

REQUIRE_ALL_LABELS_MIN_COUNT = True


# ============================================================
# リアルタイム読み取り
# ============================================================

# カメラ番号、または動画ファイルのパス
CAMERA_SOURCE = 0
CAMERA_WIDTH = None
CAMERA_HEIGHT = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CONF = 0.1
CHAR_CONF_THRES = 0.5
REALTIME_FPS = 10

TARGET_HEIGHT = 32

SHOW_WINDOW = True
SAVE_READING_LOG = True
READING_LOG_TXT = "realtime_latest_reading_log.txt"


def configure_steps():
    gemma4_labels.VIDEO_PATH = VIDEO_PATH
    gemma4_labels.YOLO_MODEL_PATH = YOLO_MODEL_PATH
    gemma4_labels.EXTRACT_DIR = IMG_DIR
    gemma4_labels.LABELS_TXT = LABELS_TXT
    gemma4_labels.TARGET_CLASSES = TARGET_CLASSES
    gemma4_labels.CLASS_VALUE_RANGES = CLASS_VALUE_RANGES
    gemma4_labels.CLASS_EXPECTED_LABEL_STEPS = CLASS_EXPECTED_LABEL_STEPS
    gemma4_labels.LABEL_MIN_COUNT = LABEL_MIN_COUNT
    gemma4_labels.LABEL_MAX_COUNT = LABEL_MAX_COUNT

    # CLASS_VALUE_RANGESから作られるので、上書き後に作り直す
    gemma4_labels.EXPECTED_LABELS = gemma4_labels.build_expected_labels()

    # 動画から直接ラベル付けし、既存の画像とラベルは消さずに追記する
    gemma4_labels.RUN_VIDEO_EXTRACTION = True
    gemma4_labels.RUN_GEMMA_EVAL = True
    gemma4_labels.CLEAR_EXTRACT_DIR = False
    gemma4_labels.RESET_LABELS_TXT = False
    gemma4_labels.APPEND_LABELS_TXT = True

    finetune_color.IMG_DIR = IMG_DIR
    finetune_color.LABELS_TXT = LABELS_TXT
    finetune_color.BASE_MODEL_PATH = BASE_MODEL_PATH
    finetune_color.FINETUNED_SAVE_DIR = FINETUNED_SAVE_DIR
    finetune_color.LATEST_MODEL_PATH = LATEST_MODEL_PATH


def print_step(title):
    print("")
    print("==================================================")
    print(title)
    print("==================================================")


def check_gemma_server():
    health_url = gemma4_labels.SERVER_URL.split("/v1/")[0] + "/health"

    try:
        requests.get(health_url, timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Gemma 4のサーバーに接続できません: {health_url}\n"
            "llama-serverを起動してから実行してください。"
        ) from e


def all_labels_min_count_reached():
    counter, _ = gemma4_labels.load_existing_label_counts(LABELS_TXT)

    gemma4_labels.print_label_count_summary(counter)

    missing = []

    for label in gemma4_labels.EXPECTED_LABELS:
        min_count, _ = gemma4_labels.get_label_limits(label)

        if min_count is not None and counter[label] < min_count:
            missing.append(label)

    print("")
    print(f"期待ラベル数: {len(gemma4_labels.EXPECTED_LABELS)}")
    print(f"不足ラベル数: {len(missing)}")

    return len(missing) == 0


def open_camera():
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if CAMERA_WIDTH is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

    if CAMERA_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: {CAMERA_SOURCE}")

    return cap


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


def load_crnn_model(model_path):
    abs_path = os.path.join(PROJECT_ROOT, model_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"モデルが見つかりません: {abs_path}")

    print(f"Loading CNN+Transformer color model: {model_path}")

    num_classes = len(ALPHABET) + 1

    model = CRNN(
        nclass=num_classes,
        nh=256,
        imgH=32,
        nc=3
    )

    state = torch.load(abs_path, map_location=DEVICE)

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

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


def predict_with_crnn(model, resized_img):
    # 学習時（cnn_transformer_color/utils.py）と同じく BGR -> RGB、[-1, 1] に正規化
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    img_norm = np.transpose(img_norm, (2, 0, 1))

    tensor = torch.from_numpy(img_norm).unsqueeze(0).to(DEVICE)

    seq_len = torch.tensor(
        [max(1, tensor.size(-1) // 2)],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        logits = model(tensor, seq_lens=seq_len)

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


def get_best_detections_by_class(results, yolo_model):
    best_by_class = {}

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
            cls_name = yolo_model.names[int(cls_id)]

            if TARGET_CLASSES is not None and cls_name not in TARGET_CLASSES:
                continue

            if cls_name not in best_by_class or float(conf) > best_by_class[cls_name]["conf"]:
                best_by_class[cls_name] = {
                    "xyxy": xyxy,
                    "cls_id": int(cls_id),
                    "cls_name": cls_name,
                    "conf": float(conf)
                }

    return best_by_class


def read_frame_with_latest_model(frame, detections, crnn_model):
    outputs = {}

    for cls_name, item in detections.items():
        x1, y1, x2, y2 = item["xyxy"].astype(int)

        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        resized = resize_keep_height(
            crop,
            target_height=TARGET_HEIGHT
        )

        if resized is None:
            continue

        text, char_confs = predict_with_crnn(crnn_model, resized)

        if not is_valid_prediction(text, char_confs, CHAR_CONF_THRES):
            continue

        outputs[cls_name] = {
            "text": text,
            "char_confs": char_confs,
            "bbox": (x1, y1, x2, y2)
        }

    return outputs


def format_reading(outputs):
    left_text = outputs.get("left", {}).get("text", "")
    right_text = outputs.get("right", {}).get("text", "")

    if left_text != "" and right_text != "":
        return f"{left_text}×10^{right_text}"

    if left_text != "":
        return f"{left_text}×10^[?]"

    if right_text != "":
        return f"[?]×10^{right_text}"

    return "有効な読み取りなし"


def draw_result(frame, detections, outputs, reading):
    vis = frame.copy()

    for cls_name, item in detections.items():
        x1, y1, x2, y2 = item["xyxy"].astype(int)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        pred = outputs.get(cls_name, {}).get("text", "")
        label = f"{cls_name}:{pred}" if pred != "" else cls_name

        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        vis,
        reading,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    return vis


def realtime_reading_with_latest_model():
    print("")
    print("========== latest.pthによるリアルタイム読み取り開始 ==========")

    if not os.path.exists(os.path.join(PROJECT_ROOT, LATEST_MODEL_PATH)):
        raise FileNotFoundError(
            f"latestモデルが見つかりません: {os.path.join(PROJECT_ROOT, LATEST_MODEL_PATH)}"
        )

    yolo_model = YOLO(os.path.join(PROJECT_ROOT, YOLO_MODEL_PATH))
    crnn_model = load_crnn_model(LATEST_MODEL_PATH)

    cap = open_camera()

    log_file = None

    if SAVE_READING_LOG:
        log_file = open(
            os.path.join(PROJECT_ROOT, READING_LOG_TXT),
            "w",
            encoding="utf-8"
        )

    frame_index = 0
    last_process_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("カメラフレームを取得できませんでした。")
                break

            now = cv2.getTickCount() / cv2.getTickFrequency()

            if now - last_process_time < 1.0 / max(REALTIME_FPS, 1e-6):
                if SHOW_WINDOW:
                    cv2.imshow("latest realtime reading", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue

            last_process_time = now
            frame_index += 1

            results = yolo_model(
                frame,
                conf=YOLO_CONF,
                verbose=False
            )

            detections = get_best_detections_by_class(results, yolo_model)

            outputs = read_frame_with_latest_model(
                frame=frame,
                detections=detections,
                crnn_model=crnn_model
            )

            reading = format_reading(outputs)

            print(f"[frame={frame_index}] {reading}")

            if log_file is not None:
                log_file.write(f"{frame_index} {reading}\n")
                log_file.flush()

            if SHOW_WINDOW:
                vis = draw_result(
                    frame=frame,
                    detections=detections,
                    outputs=outputs,
                    reading=reading
                )

                cv2.imshow("latest realtime reading", vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()

        if log_file is not None:
            log_file.close()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print("========== latest.pthによるリアルタイム読み取り終了 ==========")


def main():
    # 各スクリプトのパスはfinetune/からの相対パス
    os.chdir(PROJECT_ROOT)

    configure_steps()

    print("========== main.py start ==========")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEVICE: {DEVICE}")
    print(f"VIDEO_PATH: {VIDEO_PATH}")
    print(f"IMG_DIR: {IMG_DIR}")
    print(f"LABELS_TXT: {LABELS_TXT}")
    print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
    print(f"LATEST_MODEL_PATH: {LATEST_MODEL_PATH}")

    if RUN_GEMMA_LABELS:
        check_gemma_server()

        print_step("Gemma 4による自動ラベル付け: gemma4/gemma4_labels.py")
        gemma4_labels.main()

    if REQUIRE_ALL_LABELS_MIN_COUNT and not all_labels_min_count_reached():
        print("")
        print("全ラベルがLABEL_MIN_COUNTに到達していません。")
        print("ファインチューニングとlatestモデルでのリアルタイム読み取りは実行せずに終了します。")
        print("")
        print("対処方法:")
        print("1. VIDEO_PATHを別の動画に変えてmain.pyを再実行し、ラベルを追記で集める")
        print("2. LABEL_MIN_COUNTを下げる")
        print("3. CLASS_VALUE_RANGESを実際に出現する範囲に絞る")
        print("4. REQUIRE_ALL_LABELS_MIN_COUNT=Falseにして、今あるラベルでファインチューニングする")
        print("========== main.py finished ==========")
        return

    if RUN_FINETUNING:
        print_step("カラー版ファインチューニング: finetuning/finetune_color.py")
        finetune_color.main()

    if RUN_LATEST_MODEL_READING:
        realtime_reading_with_latest_model()

    print("")
    print("========== main.py finished ==========")


if __name__ == "__main__":
    main()
