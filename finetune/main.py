import os
import queue
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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
# ============================================================

# 表示領域（left / right）を検出するYOLO
YOLO_MODEL_PATH = "models/yolo_machine2/slip.pt"

# カメラ映像から切り出した学習画像とGemma 4のラベル
IMG_DIR = "finetuning/output_readings_camera"
LABELS_TXT = "gemma4/gemma4_labels_camera.txt"

# 学習したカラー版CNN+Transformerの保存先
# latest.pthがあれば起動時に読み込む。ゼロから学習し直す場合はフォルダを削除する
FINETUNED_SAVE_DIR = "models/camera"
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
# MIN: 全ての値がこれに届いたら学習を始める
# MAX: これに達した値はそれ以上集めない（学習後も、MAXまでは集め続ける）
LABEL_MIN_COUNT = 8
LABEL_MAX_COUNT = 30


# ============================================================
# カメラ・学習データ収集
# ============================================================

# カメラ番号、または動画ファイルのパス
CAMERA_SOURCE = 0
CAMERA_WIDTH = None
CAMERA_HEIGHT = None

# Gemma 4にラベル付けさせる間隔（秒）
COLLECT_INTERVAL_SEC = 1.0


# ============================================================
# 学習
# ============================================================

# ファインチューニングを行う間隔（分）
# 前回の学習からこの時間が経ち、新しいラベルが増えていればファインチューニングする
FINETUNE_INTERVAL_MIN = 30

# 最初の学習（ゼロから）
SCRATCH_EPOCHS = 50
SCRATCH_LR = 7e-4
SCRATCH_MIN_LR = 2e-4

# 定期的なファインチューニング（latest.pthから）
FINETUNE_EPOCHS = 30
FINETUNE_LR = 5e-4
FINETUNE_MIN_LR = 1e-4


# ============================================================
# リアルタイム読み取り
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CONF = 0.1
CHAR_CONF_THRES = 0.5
REALTIME_FPS = 10

TARGET_HEIGHT = 32

SHOW_WINDOW = True
SAVE_READING_LOG = True
READING_LOG_TXT = "realtime_latest_reading_log.txt"


def configure_steps():
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

    # 既存の画像とラベルは消さずに追記する
    gemma4_labels.RESET_LABELS_TXT = False
    gemma4_labels.APPEND_LABELS_TXT = True

    finetune_color.IMG_DIR = IMG_DIR
    finetune_color.LABELS_TXT = LABELS_TXT
    finetune_color.FINETUNED_SAVE_DIR = FINETUNED_SAVE_DIR
    finetune_color.LATEST_MODEL_PATH = LATEST_MODEL_PATH

    # 事前学習モデルは使わず、集めたデータだけで学習する
    finetune_color.BASE_MODEL_PATH = None


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


class GemmaLabelCollector:
    """切り出し画像のラベル付けを、映像を止めないよう別スレッドで行う"""

    def __init__(self):
        os.makedirs(IMG_DIR, exist_ok=True)

        label_counter, existing_files, file_mode = gemma4_labels.prepare_label_file()

        self.label_counter = label_counter
        self.existing_files = existing_files
        self.label_file = open(LABELS_TXT, file_mode, encoding="utf-8")
        self.stats = gemma4_labels.make_empty_stats()

        # メインスレッドが参照する集計。ラベル付けのたびに作り直す
        self.counts = Counter(label_counter)

        # ラベル付け中はワーカーが、学習中はメインスレッドが持つ
        self.lock = threading.Lock()

        self.idle = threading.Event()
        self.idle.set()

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, records):
        self.idle.clear()
        self.queue.put(records)

    def _run(self):
        while True:
            records = self.queue.get()

            if records is None:
                return

            for record in records:
                try:
                    with self.lock:
                        gemma4_labels.process_record_for_label(
                            record=record,
                            label_file=self.label_file,
                            label_counter=self.label_counter,
                            existing_files=self.existing_files,
                            stats=self.stats
                        )
                except Exception as e:
                    print(f"ラベル付けでエラー: {record['filename']} -> {type(e).__name__}: {e}")

                self.counts = Counter(self.label_counter)

            self.idle.set()

    def close(self):
        self.queue.put(None)
        self.thread.join()
        self.label_file.close()

        gemma4_labels.print_final_summary(self.stats, self.label_counter)


def count_ready_labels(counter):
    ready = 0

    for label in gemma4_labels.EXPECTED_LABELS:
        min_count, _ = gemma4_labels.get_label_limits(label)

        if min_count is None or counter[label] >= min_count:
            ready += 1

    return ready


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


def train_model(collector, from_scratch):
    if from_scratch:
        print_step("集まった学習データでゼロから学習: finetuning/finetune_color.py")

        finetune_color.USE_LATEST_IF_EXISTS = False
        finetune_color.EPOCHS = SCRATCH_EPOCHS
        finetune_color.LR = SCRATCH_LR
        finetune_color.MIN_LR = SCRATCH_MIN_LR
    else:
        print_step("latest.pthからファインチューニング: finetuning/finetune_color.py")

        finetune_color.USE_LATEST_IF_EXISTS = True
        finetune_color.EPOCHS = FINETUNE_EPOCHS
        finetune_color.LR = FINETUNE_LR
        finetune_color.MIN_LR = FINETUNE_MIN_LR

    # 学習中はラベル付けを止め、ラベルファイルが書き換わらないようにする
    with collector.lock:
        finetune_color.main()

    return load_crnn_model(LATEST_MODEL_PATH)


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


def crop_detection(frame, item):
    x1, y1, x2, y2 = item["xyxy"].astype(int)

    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def save_crops_for_labeling(frame, detections):
    records = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for cls_name, item in detections.items():
        crop, _ = crop_detection(frame, item)

        if crop is None:
            continue

        filename = f"camera_{timestamp}_{cls_name}.jpg"
        image_path = os.path.join(IMG_DIR, filename)

        if not cv2.imwrite(image_path, crop):
            print(f"画像保存失敗: {image_path}")
            continue

        records.append({
            "image_path": Path(image_path),
            "filename": filename,
            "yolo_class": cls_name,
            "yolo_conf": item["conf"],
            "time_sec": ""
        })

    return records


def read_frame_with_latest_model(frame, detections, crnn_model):
    outputs = {}

    for cls_name, item in detections.items():
        crop, bbox = crop_detection(frame, item)

        if crop is None:
            continue

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
            "bbox": bbox
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


def run_camera_loop(collector):
    yolo_model = YOLO(os.path.join(PROJECT_ROOT, YOLO_MODEL_PATH))

    crnn_model = None
    last_train_time = None
    labels_at_last_train = 0

    if os.path.exists(LATEST_MODEL_PATH):
        print("")
        print(f"既存のモデルを使います: {LATEST_MODEL_PATH}")
        print(f"ゼロから学習し直す場合は {FINETUNED_SAVE_DIR} を削除してください。")

        crnn_model = load_crnn_model(LATEST_MODEL_PATH)
        last_train_time = time.monotonic()
        labels_at_last_train = sum(collector.counts.values())

    expected_count = len(gemma4_labels.EXPECTED_LABELS)
    last_ready_count = None

    cap = open_camera()

    log_file = None

    if SAVE_READING_LOG:
        log_file = open(
            os.path.join(PROJECT_ROOT, READING_LOG_TXT),
            "a",
            encoding="utf-8"
        )

    frame_index = 0
    last_process_time = 0.0
    last_collect_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("カメラフレームを取得できませんでした。")
                break

            now = time.monotonic()

            if now - last_process_time < 1.0 / max(REALTIME_FPS, 1e-6):
                if SHOW_WINDOW:
                    cv2.imshow("camera", frame)

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

            # ------------------------------------------------
            # 学習データ収集（Gemma 4は別スレッド）
            # ------------------------------------------------
            counts = collector.counts

            if (
                len(detections) > 0
                and collector.idle.is_set()
                and now - last_collect_time >= COLLECT_INTERVAL_SEC
                and not gemma4_labels.is_all_expected_labels_max_reached(counts)
            ):
                records = save_crops_for_labeling(frame, detections)

                if len(records) > 0:
                    collector.submit(records)
                    last_collect_time = now

            ready_count = count_ready_labels(counts)

            if ready_count != last_ready_count:
                print(f"学習データ: {ready_count}/{expected_count} 種類の値が {LABEL_MIN_COUNT} 枚に到達")
                last_ready_count = ready_count

            # ------------------------------------------------
            # 学習
            # ------------------------------------------------
            if crnn_model is None and ready_count == expected_count:
                crnn_model = train_model(collector, from_scratch=True)
                last_train_time = time.monotonic()
                labels_at_last_train = sum(collector.counts.values())
                continue

            if (
                crnn_model is not None
                and time.monotonic() - last_train_time >= FINETUNE_INTERVAL_MIN * 60
                and sum(collector.counts.values()) > labels_at_last_train
            ):
                crnn_model = train_model(collector, from_scratch=False)
                last_train_time = time.monotonic()
                labels_at_last_train = sum(collector.counts.values())
                continue

            # ------------------------------------------------
            # 読み取り
            # ------------------------------------------------
            if crnn_model is None:
                outputs = {}
                reading = f"collecting: {ready_count}/{expected_count}"
            else:
                outputs = read_frame_with_latest_model(
                    frame=frame,
                    detections=detections,
                    crnn_model=crnn_model
                )

                reading = format_reading(outputs)

                print(f"[frame={frame_index}] {reading}")

                if log_file is not None:
                    log_file.write(f"{datetime.now().isoformat(timespec='seconds')} {reading}\n")
                    log_file.flush()

            if SHOW_WINDOW:
                vis = draw_result(
                    frame=frame,
                    detections=detections,
                    outputs=outputs,
                    reading=reading
                )

                cv2.imshow("camera", vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()

        if log_file is not None:
            log_file.close()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()


def main():
    # 各スクリプトのパスはfinetune/からの相対パス
    os.chdir(PROJECT_ROOT)

    configure_steps()

    print("========== main.py start ==========")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEVICE: {DEVICE}")
    print(f"CAMERA_SOURCE: {CAMERA_SOURCE}")
    print(f"IMG_DIR: {IMG_DIR}")
    print(f"LABELS_TXT: {LABELS_TXT}")
    print(f"LATEST_MODEL_PATH: {LATEST_MODEL_PATH}")
    print(f"LABEL_MIN_COUNT: {LABEL_MIN_COUNT}")
    print(f"LABEL_MAX_COUNT: {LABEL_MAX_COUNT}")
    print(f"FINETUNE_INTERVAL_MIN: {FINETUNE_INTERVAL_MIN}")

    check_gemma_server()

    collector = GemmaLabelCollector()

    try:
        run_camera_loop(collector)
    finally:
        collector.close()

    print("")
    print("========== main.py finished ==========")


if __name__ == "__main__":
    main()
