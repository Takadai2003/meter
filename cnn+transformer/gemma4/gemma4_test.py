import base64
import json
import mimetypes
import os
import re
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO


VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_学習用260s.mp4"
YOLO_MODEL_PATH = "models/yolo_machine2/text-best.pt"

EXTRACT_DIR = "gemma4/gemma4_test_images"
LABELS_TXT = "gemma4/gemma4_labels.txt"

TARGET_FPS = 3.75
YOLO_CONF = 0.1
GEMMA_CONF_THRES = 0.7

MAX_FRAMES = None
TARGET_CLASSES = None

RUN_VIDEO_EXTRACTION = True
RUN_GEMMA_EVAL = True
CLEAR_EXTRACT_DIR = True

USE_VALUE_RANGE_FILTER = True

DEFAULT_VALUE_MIN = None
DEFAULT_VALUE_MAX = None

CLASS_VALUE_RANGES = {
    "left": (0.0, 9.9),
    "right": (-6, 5),
}

ADD_RANGE_TO_PROMPT = True
ACCEPT_RESCUED_VALUE_WITHOUT_CONF = True

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


def image_to_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


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


def normalize_confidence(confidence):
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    return confidence


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
            return normalize_value(match.group(1))

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
            return normalize_confidence(match.group(1))

    return 0.0


def extract_json(text):
    text = text.strip()

    try:
        result = json.loads(text)
        value = normalize_value(result.get("value", None))
        confidence = normalize_confidence(result.get("confidence", 0.0))

        return {
            "value": value,
            "confidence": confidence
        }, "json_ok"
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match is not None:
        try:
            result = json.loads(match.group(0))
            value = normalize_value(result.get("value", None))
            confidence = normalize_confidence(result.get("confidence", 0.0))

            return {
                "value": value,
                "confidence": confidence
            }, "json_ok_from_text"
        except json.JSONDecodeError:
            pass

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


def ask_gemma4_with_llama_cpp(image_path, yolo_class):
    data_uri = image_to_data_uri(image_path)
    prompt = build_prompt(yolo_class)

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
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]

    result, parse_status = extract_json(raw_text)

    return result, raw_text, parse_status


def clear_extract_dir():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for path in Path(EXTRACT_DIR).iterdir():
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            path.unlink()


def get_best_detections_by_class(results, model):
    best_by_class = {}

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for xyxy, cls_id, conf in zip(boxes, cls_ids, confs):
            cls_name = model.names[int(cls_id)]

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


def extract_images_from_video():
    if CLEAR_EXTRACT_DIR:
        clear_extract_dir()
    else:
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0

    frame_interval = max(1, int(round(orig_fps / TARGET_FPS)))
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    print(f"動画: {VIDEO_PATH}")
    print(f"入力FPS: {orig_fps:.2f}")
    print(f"抽出FPS: {TARGET_FPS}")
    print(f"frame_interval: {frame_interval}")
    print(f"レンジフィルタ: {USE_VALUE_RANGE_FILTER}")
    print(f"クラス別レンジ: {CLASS_VALUE_RANGES}")

    frame_idx = 0
    processed_frames = 0
    saved_count = 0
    records = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / orig_fps
            processed_frames += 1

            results = model(
                frame,
                conf=YOLO_CONF,
                verbose=False
            )

            best_by_class = get_best_detections_by_class(results, model)

            if len(best_by_class) == 0:
                print(f"検出なし: t={time_sec:.2f}s")
            else:
                for cls_name, item in best_by_class.items():
                    x1, y1, x2, y2 = item["xyxy"]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    if crop is None or crop.size == 0:
                        continue

                    out_name = f"{video_name}_t{time_sec:06.2f}s_{cls_name}.jpg"
                    out_path = os.path.join(EXTRACT_DIR, out_name)

                    cv2.imwrite(out_path, crop)

                    record = {
                        "image_path": Path(out_path),
                        "filename": out_name,
                        "yolo_class": cls_name,
                        "yolo_conf": item["conf"],
                        "time_sec": time_sec
                    }

                    records.append(record)
                    saved_count += 1

                    print(
                        f"保存: {out_path}  "
                        f"class={cls_name}  "
                        f"conf={item['conf']:.4f}"
                    )

            if MAX_FRAMES is not None and processed_frames >= MAX_FRAMES:
                break

        frame_idx += 1

    cap.release()

    print(f"画像抽出完了: {saved_count}枚")
    print(f"処理フレーム数: {processed_frames}")
    return records


def collect_images_from_dir():
    img_dir = Path(EXTRACT_DIR)

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    records = []

    for path in paths:
        filename = path.name
        yolo_class = ""

        for class_name in CLASS_VALUE_RANGES.keys():
            if filename.endswith(f"_{class_name}.jpg"):
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


def should_accept_label(value, confidence, parse_status, yolo_class):
    if value is None:
        return False, "unreadable"

    if not is_value_in_range(value, yolo_class):
        return False, "out_of_range"

    if parse_status == "json_broken_value_rescued" and ACCEPT_RESCUED_VALUE_WITHOUT_CONF:
        return True, "accepted_rescued_value"

    if confidence < GEMMA_CONF_THRES:
        return False, "low_confidence"

    return True, "accepted"


def write_labels(records):
    os.makedirs(os.path.dirname(LABELS_TXT), exist_ok=True)

    total = 0
    saved = 0
    unreadable = 0
    low_conf = 0
    out_of_range = 0
    request_error = 0
    rescued_saved = 0

    unreadable_files = []
    low_conf_files = []
    out_of_range_files = []
    request_error_files = []
    rescued_saved_files = []

    with open(LABELS_TXT, "w", encoding="utf-8") as f:
        for record in records:
            image_path = record["image_path"]
            filename = record["filename"]
            yolo_class = record["yolo_class"]
            total += 1

            print(f"Gemma 4評価中: {image_path}")

            try:
                result, raw_text, parse_status = ask_gemma4_with_llama_cpp(
                    image_path=image_path,
                    yolo_class=yolo_class
                )
            except Exception as e:
                request_error += 1
                request_error_files.append((filename, str(e)))

                print(
                    f"除外: {filename}  "
                    f"reason=request_error  error={e}"
                )
                continue

            value = result["value"]
            confidence = result["confidence"]

            accepted, reason = should_accept_label(
                value=value,
                confidence=confidence,
                parse_status=parse_status,
                yolo_class=yolo_class
            )

            if accepted:
                f.write(f"{filename} {value}\n")
                saved += 1

                if reason == "accepted_rescued_value":
                    rescued_saved += 1
                    rescued_saved_files.append((filename, value, confidence, parse_status))

                print(
                    f"採用: {filename}  "
                    f"value={value}  "
                    f"conf={confidence:.2f}  "
                    f"parse={parse_status}  "
                    f"reason={reason}"
                )
                continue

            if reason == "unreadable":
                unreadable += 1
                unreadable_files.append((filename, parse_status))

            elif reason == "out_of_range":
                out_of_range += 1
                out_of_range_files.append((filename, value, confidence, yolo_class, parse_status))

            elif reason == "low_confidence":
                low_conf += 1
                low_conf_files.append((filename, value, confidence, parse_status))

            print(
                f"除外: {filename}  "
                f"reason={reason}  "
                f"value={value}  "
                f"conf={confidence:.2f}  "
                f"parse={parse_status}  "
                f"class={yolo_class}"
            )

    rejected = unreadable + low_conf + out_of_range + request_error

    print("")
    print("Gemma 4ラベル生成完了")
    print(f"評価枚数: {total}")
    print(f"採用枚数: {saved}")
    print(f"除外枚数: {rejected}")
    print(f"json_broken_value_rescued から採用した枚数: {rescued_saved}")
    print(f"Gemma 4が読み取れなかった画像枚数: {unreadable}")
    print(f"Gemma 4のconfidence閾値未満でラベル付けされなかった画像枚数: {low_conf}")
    print(f"レンジ外として除外した画像枚数: {out_of_range}")
    print(f"Gemma 4問い合わせエラー枚数: {request_error}")
    print(f"保存先: {LABELS_TXT}")

    print("")
    print("レンジ外として除外したファイル名")
    if len(out_of_range_files) == 0:
        print("なし")
    else:
        for filename, value, confidence, yolo_class, parse_status in out_of_range_files:
            print(
                f"{filename}  "
                f"class={yolo_class}  "
                f"value={value}  "
                f"conf={confidence:.2f}  "
                f"parse={parse_status}"
            )

    print("")
    print("Gemma 4が読み取れなかったファイル名")
    if len(unreadable_files) == 0:
        print("なし")
    else:
        for filename, parse_status in unreadable_files:
            print(f"{filename}  parse={parse_status}")

    print("")
    print(f"confidence < {GEMMA_CONF_THRES} のためラベル付けされなかったファイル名")
    if len(low_conf_files) == 0:
        print("なし")
    else:
        for filename, value, confidence, parse_status in low_conf_files:
            print(
                f"{filename}  "
                f"value={value}  "
                f"conf={confidence:.2f}  "
                f"parse={parse_status}"
            )

    print("")
    print("json_broken_value_rescued から採用したファイル名")
    if len(rescued_saved_files) == 0:
        print("なし")
    else:
        for filename, value, confidence, parse_status in rescued_saved_files:
            print(
                f"{filename}  "
                f"value={value}  "
                f"conf={confidence:.2f}  "
                f"parse={parse_status}"
            )

    print("")
    print("Gemma 4問い合わせエラーのファイル名")
    if len(request_error_files) == 0:
        print("なし")
    else:
        for filename, error in request_error_files:
            print(f"{filename}  error={error}")


def main():
    if RUN_VIDEO_EXTRACTION:
        records = extract_images_from_video()
    else:
        records = collect_images_from_dir()

    if len(records) == 0:
        print("評価対象画像がありません")
        return

    if RUN_GEMMA_EVAL:
        write_labels(records)
    else:
        print("RUN_GEMMA_EVAL=False のため、Gemma 4評価は実行しません")


if __name__ == "__main__":
    main()