import base64
import json
import mimetypes
import os
import re
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO


# ============================================================
# 設定
# ============================================================

VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_slip_right.mp4"
YOLO_MODEL_PATH = "models/yolo_machine2/slip.pt"

EXTRACT_DIR = "finetuning/output_readings_slip"
LABELS_TXT = "gemma4/gemma4_labels_slip_right.txt"

TARGET_FPS = 20
YOLO_CONF = 0.1
GEMMA_CONF_THRES = 0.5

MAX_FRAMES = None

# 処理対象YOLOクラス
TARGET_CLASSES = ["right"]

RUN_VIDEO_EXTRACTION = True
RUN_GEMMA_EVAL = True
CLEAR_EXTRACT_DIR = False

RESET_LABELS_TXT = False
APPEND_LABELS_TXT = True


# ============================================================
# デバッグ表示
# ============================================================

# TrueにするとGemma 4の生の回答を表示
PRINT_RAW_GEMMA_RESPONSE = True


# ============================================================
# 値レンジ
# ============================================================

USE_VALUE_RANGE_FILTER = True

DEFAULT_VALUE_MIN = None
DEFAULT_VALUE_MAX = None

CLASS_VALUE_RANGES = {
    "left": (1.0, 9.9),
    "right": (-5, 5),
}

CLASS_EXPECTED_LABEL_STEPS = {
    "left": 0.1,
    "right": 1,
}

ADD_RANGE_TO_PROMPT = True

# JSONが壊れていてもvalueを救出できた場合は
# confidenceに関係なく採用
ACCEPT_RESCUED_VALUE_WITHOUT_CONF = True


# ============================================================
# ラベル枚数制限
# ============================================================

USE_LABEL_COUNT_FILTER = True

LABEL_MIN_COUNT = 8
LABEL_MAX_COUNT = 10

# 個別指定例:
# LABEL_LIMITS = {
#     "1.0": (8, 12),
#     "1.1": (8, 15),
# }
LABEL_LIMITS = {}

STOP_WHEN_ALL_LABELS_MAX_REACHED = True


# ============================================================
# EXPECTED_LABELS作成
# ============================================================

def build_expected_labels():
    expected_labels = []
    seen = set()

    for class_name, value_range in CLASS_VALUE_RANGES.items():
        if value_range is None or len(value_range) != 2:
            continue

        min_value, max_value = value_range
        step = CLASS_EXPECTED_LABEL_STEPS.get(class_name)

        if min_value is None or max_value is None or step is None:
            continue

        try:
            d_min = Decimal(str(min_value))
            d_max = Decimal(str(max_value))
            d_step = Decimal(str(step))
        except Exception:
            continue

        if d_step <= 0:
            continue

        if d_step == d_step.to_integral_value():
            d_step = d_step.to_integral_value()

        if d_min > d_max:
            d_min, d_max = d_max, d_min

        start_n = int(
            (d_min / d_step).to_integral_value(
                rounding=ROUND_CEILING
            )
        )

        end_n = int(
            (d_max / d_step).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )

        if start_n > end_n:
            continue

        decimals = max(
            0,
            -d_step.as_tuple().exponent
        )

        quant = Decimal("1").scaleb(-decimals)

        for n in range(start_n, end_n + 1):
            value = (d_step * n).quantize(quant)

            if decimals == 0:
                label = str(int(value))
            else:
                label = f"{value:.{decimals}f}"

            if label in seen:
                continue

            expected_labels.append(label)
            seen.add(label)

    return expected_labels


EXPECTED_LABELS = build_expected_labels()


# ============================================================
# Gemma 4
# ============================================================

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "gemma4"

BASE_PROMPT = """
あなたは計測器のデジタル表示を読む専用のOCRアシスタントです。
画像に写っている計測器の数値表示を読んでください。

出力は次のJSON形式にしてください。
読み取りにくい場合でも信頼度を下げて数値をできるだけ出力してください。

{
  "value": "読み取った数値",
  "confidence": "読み取りの自信度（0.0から1.0の数値）"
}

計測器の条件:
- 計測器の数値は緑色で表示されています。
- 7セグメント表示、小数点、マイナス記号に注意してください。
- valueの候補は以下の辞書から選んでください。
    辞書 = [0,1,2,3,4,5,6,7,8,9,"-","."]
"""


# ============================================================
# Range
# ============================================================

def get_value_range(yolo_class):
    if yolo_class in CLASS_VALUE_RANGES:
        return CLASS_VALUE_RANGES[yolo_class]

    if DEFAULT_VALUE_MIN is None and DEFAULT_VALUE_MAX is None:
        return None

    return DEFAULT_VALUE_MIN, DEFAULT_VALUE_MAX


def build_prompt(yolo_class):
    prompt = BASE_PROMPT

    if not ADD_RANGE_TO_PROMPT:
        return prompt

    value_range = get_value_range(yolo_class)

    if value_range is None:
        return prompt

    min_value, max_value = value_range

    prompt += f"""
追加条件:
- この画像はYOLOで「{yolo_class}」クラスとして検出された領域です。
- このクラスの計測値は {min_value} 以上 {max_value} 以下の範囲です。
- valueは必ずこの範囲内の数値として出力してください。
"""

    return prompt


# ============================================================
# 画像 → Data URI
# ============================================================

def image_to_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# Value処理
# ============================================================

def normalize_value(value):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace("−", "-")
    value = re.sub(r"[^0-9.\-]", "", value)

    if value == "":
        return None

    if value.count(".") > 1:
        return None

    if value.count("-") > 1:
        return None

    if "-" in value and not value.startswith("-"):
        return None

    return value


def value_to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def is_value_in_range(value, yolo_class):
    if not USE_VALUE_RANGE_FILTER:
        return True

    value_range = get_value_range(yolo_class)

    if value_range is None:
        return True

    value_float = value_to_float(value)

    if value_float is None:
        return False

    min_value, max_value = value_range

    if min_value is not None and value_float < min_value:
        return False

    if max_value is not None and value_float > max_value:
        return False

    return True


# ============================================================
# Confidence
# ============================================================

def normalize_confidence(confidence):
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    return max(
        0.0,
        min(1.0, confidence)
    )


# ============================================================
# ラベル枚数
# ============================================================

def get_label_limits(label):
    if label in LABEL_LIMITS:
        return LABEL_LIMITS[label]

    return LABEL_MIN_COUNT, LABEL_MAX_COUNT


def is_label_count_acceptable(label, counter):
    if not USE_LABEL_COUNT_FILTER:
        return True

    _, max_count = get_label_limits(label)

    if max_count is None:
        return True

    return counter[label] < max_count


def is_all_expected_labels_max_reached(counter):
    if not STOP_WHEN_ALL_LABELS_MAX_REACHED:
        return False

    if not USE_LABEL_COUNT_FILTER:
        return False

    if EXPECTED_LABELS is None:
        return False

    if len(EXPECTED_LABELS) == 0:
        return False

    checked_count = 0

    for label in EXPECTED_LABELS:
        _, max_count = get_label_limits(label)

        if max_count is None:
            continue

        checked_count += 1

        if counter[label] < max_count:
            return False

    if checked_count == 0:
        return False

    return True


# ============================================================
# Gemma 4出力解析
# ============================================================

def extract_value_from_text(text):
    patterns = [
        r'"value"\s*:\s*"([^"]*)"',
        r"'value'\s*:\s*'([^']*)'",
        r'"value"\s*:\s*([\-0-9.]+)',
        r"'value'\s*:\s*([\-0-9.]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return normalize_value(
                match.group(1)
            )

    return None


def extract_confidence_from_text(text):
    patterns = [
        r'"confidence"\s*:\s*"([^"]*)"',
        r"'confidence'\s*:\s*'([^']*)'",
        r'"confidence"\s*:\s*([0-9.]+)',
        r"'confidence'\s*:\s*([0-9.]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return normalize_confidence(
                match.group(1)
            )

    return 0.0


def extract_json(text):
    text = text.strip()

    # 完全なJSON
    try:
        result = json.loads(text)

        value = normalize_value(
            result.get("value")
        )

        confidence = normalize_confidence(
            result.get("confidence", 0.0)
        )

        return {
            "value": value,
            "confidence": confidence
        }, "json_ok"

    except json.JSONDecodeError:
        pass

    # 文章中のJSON
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match is not None:
        try:
            result = json.loads(
                match.group(0)
            )

            value = normalize_value(
                result.get("value")
            )

            confidence = normalize_confidence(
                result.get("confidence", 0.0)
            )

            return {
                "value": value,
                "confidence": confidence
            }, "json_ok_from_text"

        except json.JSONDecodeError:
            pass

    # JSONが壊れていてもvalueを救出
    value = extract_value_from_text(text)
    confidence = extract_confidence_from_text(text)

    if value is not None:
        return {
            "value": value,
            "confidence": confidence
        }, "json_broken_value_rescued"

    return {
        "value": None,
        "confidence": 0.0
    }, "json_failed"


# ============================================================
# Gemma 4問い合わせ
# ============================================================

def ask_gemma4_with_llama_cpp(
    image_path,
    yolo_class
):
    data_uri = image_to_data_uri(
        image_path
    )

    prompt = build_prompt(
        yolo_class
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 512,
        "response_format": {
            "type": "json_object"
        }
    }

    response = requests.post(
        SERVER_URL,
        headers={
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    raw_text = (
        data["choices"][0]
        ["message"]["content"]
    )

    result, parse_status = extract_json(
        raw_text
    )

    return result, raw_text, parse_status


# ============================================================
# 出力画像削除
# ============================================================

def clear_extract_dir():
    os.makedirs(
        EXTRACT_DIR,
        exist_ok=True
    )

    for path in Path(EXTRACT_DIR).iterdir():
        if (
            path.is_file()
            and path.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]
        ):
            path.unlink()


# ============================================================
# YOLO
# ============================================================

def get_best_detections_by_class(
    results,
    model
):
    best_by_class = {}

    for result in results:
        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
            .astype(int)
        )

        cls_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confs = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for xyxy, cls_id, conf in zip(
            boxes,
            cls_ids,
            confs
        ):
            cls_name = model.names[
                int(cls_id)
            ]

            if (
                TARGET_CLASSES is not None
                and cls_name not in TARGET_CLASSES
            ):
                continue

            if (
                cls_name not in best_by_class
                or float(conf)
                > best_by_class[cls_name]["conf"]
            ):
                best_by_class[cls_name] = {
                    "xyxy": xyxy,
                    "cls_id": int(cls_id),
                    "cls_name": cls_name,
                    "conf": float(conf)
                }

    return best_by_class


# ============================================================
# 既存ラベル
# ============================================================

def load_existing_label_counts(
    labels_txt
):
    counter = Counter()
    existing_files = set()

    if not os.path.exists(labels_txt):
        return counter, existing_files

    with open(
        labels_txt,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            parts = line.split(
                maxsplit=1
            )

            if len(parts) < 2:
                continue

            filename = parts[0]
            label = parts[1].strip()

            counter[label] += 1
            existing_files.add(filename)

    return counter, existing_files


# ============================================================
# 採用判定
# ============================================================

def should_accept_label(
    value,
    confidence,
    parse_status,
    yolo_class,
    label_counter
):
    if value is None:
        return False, "unreadable"

    if not is_value_in_range(
        value,
        yolo_class
    ):
        return False, "out_of_range"

    if not is_label_count_acceptable(
        value,
        label_counter
    ):
        return False, "over_max_count"

    if (
        parse_status
        == "json_broken_value_rescued"
        and ACCEPT_RESCUED_VALUE_WITHOUT_CONF
    ):
        return True, "accepted_rescued_value"

    if confidence < GEMMA_CONF_THRES:
        return False, "low_confidence"

    return True, "accepted"


# ============================================================
# ラベル数表示
# ============================================================

def print_label_count_summary(counter):
    print("")
    print("========== ラベル数集計 ==========")

    total = sum(
        counter.values()
    )

    print(
        f"総採用サンプル数: {total}"
    )

    def sort_key(item):
        label = item[0]

        try:
            return (
                0,
                float(label)
            )
        except Exception:
            return (
                1,
                label
            )

    for label, count in sorted(
        counter.items(),
        key=sort_key
    ):
        min_count, max_count = (
            get_label_limits(label)
        )

        status = []

        if (
            min_count is not None
            and count < min_count
        ):
            status.append(
                "below_min"
            )

        if (
            max_count is not None
            and count >= max_count
        ):
            status.append(
                "max_reached"
            )

        status_text = (
            ",".join(status)
            if len(status) > 0
            else "ok"
        )

        print(
            f"{label:>8} : {count:>5}  "
            f"min={min_count}  "
            f"max={max_count}  "
            f"{status_text}"
        )

    if EXPECTED_LABELS is not None:
        print("")
        print(
            "========== 下限未満の期待ラベル =========="
        )

        shortage_count = 0

        for label in EXPECTED_LABELS:
            count = counter[label]
            min_count, _ = (
                get_label_limits(label)
            )

            if (
                min_count is not None
                and count < min_count
            ):
                shortage_count += 1

                print(
                    f"{label:>8} : "
                    f"{count} / {min_count}"
                )

        if shortage_count == 0:
            print("なし")


# ============================================================
# Stats
# ============================================================

def make_empty_stats():
    return {
        "processed_frames": 0,
        "extracted_images": 0,
        "eval_total": 0,
        "saved": 0,
        "unreadable": 0,
        "low_conf": 0,
        "out_of_range": 0,
        "over_max": 0,
        "request_error": 0,
        "rescued_saved": 0,
        "duplicated": 0,
        "early_stopped": False,

        "unreadable_files": [],
        "low_conf_files": [],
        "out_of_range_files": [],
        "over_max_files": [],
        "request_error_files": [],
        "rescued_saved_files": [],
        "duplicated_files": [],
    }


# ============================================================
# 1画像をGemma 4で処理
# ============================================================

def process_record_for_label(
    record,
    label_file,
    label_counter,
    existing_files,
    stats
):
    image_path = record["image_path"]
    filename = record["filename"]
    yolo_class = record["yolo_class"]

    stats["eval_total"] += 1

    # ========================================================
    # 重複
    # ========================================================

    if filename in existing_files:
        stats["duplicated"] += 1

        stats["duplicated_files"].append(
            filename
        )

        print(
            f"除外: {filename}  "
            f"reason=duplicated"
        )

        # ラベル付け済みの画像なので削除しない（ファインチューニングで使う）
        return

    print(
        f"Gemma 4評価中: {image_path}"
    )

    # ========================================================
    # Gemma 4問い合わせ
    # ========================================================

    try:
        result, raw_text, parse_status = (
            ask_gemma4_with_llama_cpp(
                image_path=image_path,
                yolo_class=yolo_class
            )
        )

    except Exception as e:
        stats["request_error"] += 1

        stats["request_error_files"].append(
            (
                filename,
                str(e)
            )
        )

        # ★ エラー内容を表示
        print(
            f"Gemma 4エラー: {filename} "
            f"-> {type(e).__name__}: {e}"
        )

        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

        return

    # ========================================================
    # 生のGemma 4出力
    # ========================================================

    if PRINT_RAW_GEMMA_RESPONSE:
        print(
            f"Gemma 4 raw: {raw_text}"
        )

    value = result["value"]
    confidence = result["confidence"]

    # Gemma 4の解析結果も表示
    print(
        f"Gemma 4 result: "
        f"value={value}  "
        f"conf={confidence:.2f}  "
        f"parse={parse_status}"
    )

    # ========================================================
    # 採用判定
    # ========================================================

    accepted, reason = should_accept_label(
        value=value,
        confidence=confidence,
        parse_status=parse_status,
        yolo_class=yolo_class,
        label_counter=label_counter
    )

    # ========================================================
    # 採用
    # ========================================================

    if accepted:
        label_file.write(
            f"{filename} {value}\n"
        )

        label_file.flush()

        label_counter[value] += 1
        existing_files.add(filename)

        stats["saved"] += 1

        if (
            reason
            == "accepted_rescued_value"
        ):
            stats["rescued_saved"] += 1

            stats[
                "rescued_saved_files"
            ].append(
                (
                    filename,
                    value,
                    confidence,
                    parse_status
                )
            )

        print(
            f"採用: {filename}  "
            f"value={value}  "
            f"count={label_counter[value]}  "
            f"conf={confidence:.2f}  "
            f"parse={parse_status}  "
            f"reason={reason}"
        )

        return

    # ========================================================
    # 不採用
    # ========================================================

    # ★ ここで原因を必ず表示
    print(
        f"除外: {filename}  "
        f"value={value}  "
        f"conf={confidence:.2f}  "
        f"parse={parse_status}  "
        f"reason={reason}"
    )

    # 不採用画像を削除
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass

    # ========================================================
    # 理由別カウント
    # ========================================================

    if reason == "unreadable":
        stats["unreadable"] += 1

        stats[
            "unreadable_files"
        ].append(
            (
                filename,
                parse_status
            )
        )

    elif reason == "out_of_range":
        stats["out_of_range"] += 1

        stats[
            "out_of_range_files"
        ].append(
            (
                filename,
                value,
                confidence,
                yolo_class,
                parse_status
            )
        )

    elif reason == "low_confidence":
        stats["low_conf"] += 1

        stats[
            "low_conf_files"
        ].append(
            (
                filename,
                value,
                confidence,
                parse_status
            )
        )

    elif reason == "over_max_count":
        stats["over_max"] += 1

        stats[
            "over_max_files"
        ].append(
            (
                filename,
                value,
                confidence,
                label_counter[value],
                parse_status
            )
        )


# ============================================================
# 最終集計
# ============================================================

def print_final_summary(
    stats,
    label_counter
):
    rejected = (
        stats["unreadable"]
        + stats["low_conf"]
        + stats["out_of_range"]
        + stats["over_max"]
        + stats["request_error"]
        + stats["duplicated"]
    )

    print("")
    print("Gemma 4ラベル生成完了")

    print(
        f"処理フレーム数: "
        f"{stats['processed_frames']}"
    )

    print(
        f"抽出画像数: "
        f"{stats['extracted_images']}"
    )

    print(
        f"評価枚数: "
        f"{stats['eval_total']}"
    )

    print(
        f"採用枚数: "
        f"{stats['saved']}"
    )

    print(
        f"除外枚数: "
        f"{rejected}"
    )

    print(
        f"読み取れなかった枚数: "
        f"{stats['unreadable']}"
    )

    print(
        f"confidence閾値未満枚数: "
        f"{stats['low_conf']}"
    )

    print(
        f"レンジ外枚数: "
        f"{stats['out_of_range']}"
    )

    print(
        f"ラベル上限到達で除外した枚数: "
        f"{stats['over_max']}"
    )

    print(
        f"json_broken_value_rescuedから採用した枚数: "
        f"{stats['rescued_saved']}"
    )

    print(
        f"重複ファイル名で除外した枚数: "
        f"{stats['duplicated']}"
    )

    print(
        f"Gemma 4問い合わせエラー枚数: "
        f"{stats['request_error']}"
    )

    print(
        f"全ラベル上限到達による早期停止: "
        f"{stats['early_stopped']}"
    )

    print(
        f"保存先: {LABELS_TXT}"
    )

    print_label_count_summary(
        label_counter
    )


# ============================================================
# ラベルファイル準備
# ============================================================

def prepare_label_file():
    label_dir = os.path.dirname(
        LABELS_TXT
    )

    if label_dir:
        os.makedirs(
            label_dir,
            exist_ok=True
        )

    if (
        APPEND_LABELS_TXT
        and not RESET_LABELS_TXT
    ):
        label_counter, existing_files = (
            load_existing_label_counts(
                LABELS_TXT
            )
        )

        file_mode = "a"

    else:
        label_counter = Counter()
        existing_files = set()
        file_mode = "w"

    return (
        label_counter,
        existing_files,
        file_mode
    )


# ============================================================
# 動画から直接処理
# ============================================================

def process_video_streaming():
    if CLEAR_EXTRACT_DIR:
        clear_extract_dir()
    else:
        os.makedirs(
            EXTRACT_DIR,
            exist_ok=True
        )

    # YOLO
    model = YOLO(
        YOLO_MODEL_PATH
    )

    # 動画
    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"動画を開けませんでした: "
            f"{VIDEO_PATH}"
        )

    orig_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0

    frame_interval = max(
        1,
        int(
            round(
                orig_fps / TARGET_FPS
            )
        )
    )

    actual_fps = (
        orig_fps / frame_interval
    )

    video_name = os.path.splitext(
        os.path.basename(
            VIDEO_PATH
        )
    )[0]

    print(
        f"動画: {VIDEO_PATH}"
    )

    print(
        f"入力FPS: {orig_fps:.2f}"
    )

    print(
        f"指定抽出FPS: {TARGET_FPS}"
    )

    print(
        f"実際の抽出FPS: {actual_fps:.3f}"
    )

    print(
        f"frame_interval: {frame_interval}"
    )

    print(
        f"処理対象クラス: {TARGET_CLASSES}"
    )

    print(
        f"レンジフィルタ: "
        f"{USE_VALUE_RANGE_FILTER}"
    )

    print(
        f"クラス別レンジ: "
        f"{CLASS_VALUE_RANGES}"
    )

    print(
        f"Gemma confidence閾値: "
        f"{GEMMA_CONF_THRES}"
    )

    print(
        f"ラベル数制限: "
        f"{USE_LABEL_COUNT_FILTER}"
    )

    print(
        f"LABEL_MIN_COUNT: "
        f"{LABEL_MIN_COUNT}"
    )

    print(
        f"LABEL_MAX_COUNT: "
        f"{LABEL_MAX_COUNT}"
    )

    print(
        f"LABEL_LIMITS: "
        f"{LABEL_LIMITS}"
    )

    print(
        f"EXPECTED_LABELS数: "
        f"{len(EXPECTED_LABELS)}"
    )

    print(
        f"全ラベル上限到達で停止: "
        f"{STOP_WHEN_ALL_LABELS_MAX_REACHED}"
    )

    print(
        f"Gemma raw表示: "
        f"{PRINT_RAW_GEMMA_RESPONSE}"
    )

    print("")

    label_counter, existing_files, file_mode = (
        prepare_label_file()
    )

    stats = make_empty_stats()

    label_file = None

    try:
        if RUN_GEMMA_EVAL:
            label_file = open(
                LABELS_TXT,
                file_mode,
                encoding="utf-8"
            )

        frame_idx = 0

        while True:
            # 全ラベルが上限に到達
            if (
                RUN_GEMMA_EVAL
                and is_all_expected_labels_max_reached(
                    label_counter
                )
            ):
                stats["early_stopped"] = True

                print("")
                print(
                    "全EXPECTED_LABELSが"
                    "LABEL_MAX_COUNTに到達したため、"
                    "動画抽出とGemma 4推論を停止します。"
                )

                break

            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # TARGET_FPS相当のフレームだけ処理
            if (
                frame_idx
                % frame_interval
                != 0
            ):
                continue

            time_sec = (
                frame_idx / orig_fps
            )

            stats["processed_frames"] += 1

            # =================================================
            # YOLO
            # =================================================

            results = model(
                frame,
                conf=YOLO_CONF,
                verbose=False
            )

            best_by_class = (
                get_best_detections_by_class(
                    results,
                    model
                )
            )

            if len(best_by_class) == 0:
                print(
                    f"検出なし: "
                    f"t={time_sec:.2f}s"
                )

            else:
                for cls_name, item in (
                    best_by_class.items()
                ):
                    if (
                        RUN_GEMMA_EVAL
                        and
                        is_all_expected_labels_max_reached(
                            label_counter
                        )
                    ):
                        stats[
                            "early_stopped"
                        ] = True

                        print("")
                        print(
                            "全EXPECTED_LABELSが"
                            "LABEL_MAX_COUNTに到達したため、"
                            "動画抽出とGemma 4推論を停止します。"
                        )

                        break

                    x1, y1, x2, y2 = (
                        item["xyxy"]
                    )

                    x1 = max(
                        0,
                        x1
                    )

                    y1 = max(
                        0,
                        y1
                    )

                    x2 = min(
                        frame.shape[1],
                        x2
                    )

                    y2 = min(
                        frame.shape[0],
                        y2
                    )

                    if (
                        x2 <= x1
                        or y2 <= y1
                    ):
                        continue

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if (
                        crop is None
                        or crop.size == 0
                    ):
                        continue

                    out_name = (
                        f"{video_name}_"
                        f"t{time_sec:06.2f}s_"
                        f"{cls_name}.jpg"
                    )

                    out_path = os.path.join(
                        EXTRACT_DIR,
                        out_name
                    )

                    success = cv2.imwrite(
                        out_path,
                        crop
                    )

                    if not success:
                        print(
                            f"画像保存失敗: "
                            f"{out_path}"
                        )
                        continue

                    stats[
                        "extracted_images"
                    ] += 1

                    print(
                        f"保存: {out_path}  "
                        f"class={cls_name}  "
                        f"conf={item['conf']:.4f}"
                    )

                    # =================================================
                    # Gemma 4
                    # =================================================

                    if RUN_GEMMA_EVAL:
                        record = {
                            "image_path": Path(
                                out_path
                            ),
                            "filename": out_name,
                            "yolo_class": cls_name,
                            "yolo_conf": item[
                                "conf"
                            ],
                            "time_sec": time_sec
                        }

                        process_record_for_label(
                            record=record,
                            label_file=label_file,
                            label_counter=label_counter,
                            existing_files=existing_files,
                            stats=stats
                        )

                if stats["early_stopped"]:
                    break

            if (
                MAX_FRAMES is not None
                and
                stats["processed_frames"]
                >= MAX_FRAMES
            ):
                break

    finally:
        cap.release()

        if label_file is not None:
            label_file.close()

    if RUN_GEMMA_EVAL:
        print_final_summary(
            stats,
            label_counter
        )

    else:
        print("")
        print("動画抽出完了")

        print(
            f"処理フレーム数: "
            f"{stats['processed_frames']}"
        )

        print(
            f"抽出画像数: "
            f"{stats['extracted_images']}"
        )


# ============================================================
# 既存画像を読み込む
# ============================================================

def collect_images_from_dir():
    img_dir = Path(
        EXTRACT_DIR
    )

    if not img_dir.exists():
        raise FileNotFoundError(
            img_dir
        )

    paths = sorted([
        p
        for p in img_dir.iterdir()
        if p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]
    ])

    records = []

    for path in paths:
        filename = path.name
        stem = path.stem
        yolo_class = ""

        for class_name in (
            CLASS_VALUE_RANGES.keys()
        ):
            if stem.endswith(
                f"_{class_name}"
            ):
                yolo_class = class_name
                break

        records.append({
            "image_path": path,
            "filename": filename,
            "yolo_class": yolo_class,
            "yolo_conf": "",
            "time_sec": ""
        })

    return records


# ============================================================
# 既存画像をGemma 4で評価
# ============================================================

def write_labels_from_records(
    records
):
    label_counter, existing_files, file_mode = (
        prepare_label_file()
    )

    stats = make_empty_stats()

    with open(
        LABELS_TXT,
        file_mode,
        encoding="utf-8"
    ) as label_file:

        for record in records:
            if is_all_expected_labels_max_reached(
                label_counter
            ):
                stats[
                    "early_stopped"
                ] = True

                print("")
                print(
                    "全EXPECTED_LABELSが"
                    "LABEL_MAX_COUNTに到達したため、"
                    "Gemma 4推論を停止します。"
                )

                break

            process_record_for_label(
                record=record,
                label_file=label_file,
                label_counter=label_counter,
                existing_files=existing_files,
                stats=stats
            )

            if is_all_expected_labels_max_reached(
                label_counter
            ):
                stats[
                    "early_stopped"
                ] = True

                print("")
                print(
                    "全EXPECTED_LABELSが"
                    "LABEL_MAX_COUNTに到達したため、"
                    "Gemma 4推論を停止します。"
                )

                break

    print_final_summary(
        stats,
        label_counter
    )


# ============================================================
# Main
# ============================================================

def main():
    if RUN_VIDEO_EXTRACTION:
        process_video_streaming()
        return

    records = collect_images_from_dir()

    if len(records) == 0:
        print(
            "評価対象画像がありません"
        )
        return

    if RUN_GEMMA_EVAL:
        write_labels_from_records(
            records
        )

    else:
        print(
            "RUN_GEMMA_EVAL=False のため、"
            "Gemma 4評価は実行しません"
        )


if __name__ == "__main__":
    main()