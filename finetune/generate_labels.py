import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from cnn_transformer.cnn_transformer import CRNN
from cnn_transformer.utils import ALPHABET, indices_to_text


# ============================================================
# パス設定
# ============================================================
VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_dark_val.mp4"

YOLO_MODEL = "models/yolo_machine2/best.pt"
CRNN_MODEL = "models/transformer_color/best-model_color_4.pth"

OUTPUT_DIR = "finetuning/output_readings_dark/val"
OUTPUT_LABEL_TXT = "finetuning/output_readings_dark/val_labels.txt"


# ============================================================
# デバイス
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 動画・YOLO設定
# ============================================================
TARGET_FPS = 2
CONF_THRES = 0.1


# ============================================================
# 処理・保存するYOLOクラス
#
# ["left"]          → leftだけ処理・保存
# ["right"]         → rightだけ処理・保存
# ["left", "right"] → 両方処理・保存
# None              → 全クラス処理・保存
# ============================================================
TARGET_CLASSES = ["left"]


# ============================================================
# CRNN入力サイズ
# ============================================================
TARGET_HEIGHT = 32


# True  = ラベルファイルへ追記
# False = ラベルファイルを上書き
APPEND_LABEL_FILE = False


# ============================================================
# 高さ32固定・アスペクト比維持リサイズ
# ============================================================
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


# ============================================================
# CRNN用前処理
#
# カラー画像のまま高さ32にするだけ
# ============================================================
def preprocess_for_crnn(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    resized = resize_keep_height(
        crop_bgr,
        target_height=TARGET_HEIGHT
    )

    return resized


# ============================================================
# CRNNモデル読み込み
# ============================================================
def load_crnn_model():
    print("Loading CRNN model...")
    print(f"CRNN model: {CRNN_MODEL}")
    print(f"Device: {DEVICE}")

    num_classes = len(ALPHABET) + 1

    # カラー3チャンネル
    crnn = CRNN(
        nclass=num_classes,
        nh=256,
        imgH=32,
        nc=3
    )

    state = torch.load(
        CRNN_MODEL,
        map_location=DEVICE
    )

    crnn.load_state_dict(state)
    crnn.to(DEVICE)
    crnn.eval()

    return crnn


# ============================================================
# CRNN推論
# ============================================================
def predict_text(crnn, img_color):

    # [0,255] → [0,1]
    img_norm = img_color.astype(np.float32) / 255.0

    # [0,1] → [-1,1]
    img_norm = (img_norm - 0.5) / 0.5

    # OpenCV:
    # [H, W, C]
    #
    # PyTorch:
    # [C, H, W]
    img_norm = np.transpose(
        img_norm,
        (2, 0, 1)
    )

    # [C,H,W] → [1,C,H,W]
    tensor = (
        torch.from_numpy(img_norm)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = crnn(tensor)

        logits = logits.permute(
            1,
            0,
            2
        )

        preds = (
            logits
            .softmax(2)
            .argmax(2)[0]
            .cpu()
            .numpy()
            .tolist()
        )

        text = indices_to_text(preds)

    return text.strip()


# ============================================================
# メイン
# ============================================================
def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    label_dir = os.path.dirname(
        OUTPUT_LABEL_TXT
    )

    if label_dir:
        os.makedirs(
            label_dir,
            exist_ok=True
        )


    # ========================================================
    # YOLO読み込み
    # ========================================================
    print("Loading YOLO model...")

    yolo = YOLO(
        YOLO_MODEL
    )


    # ========================================================
    # YOLOクラス一覧
    # ========================================================
    print("")
    print("YOLO classes:")

    for class_id, class_name in yolo.names.items():
        print(
            f"  {class_id}: {class_name}"
        )


    # ========================================================
    # 対象クラスチェック
    # ========================================================
    print("")

    if TARGET_CLASSES is None:
        print("処理・保存対象クラス: 全クラス")

    else:
        print(
            f"処理・保存対象クラス: {TARGET_CLASSES}"
        )

        available_classes = set(
            yolo.names.values()
        )

        invalid_classes = [
            name
            for name in TARGET_CLASSES
            if name not in available_classes
        ]

        if invalid_classes:
            print(
                "存在しないYOLOクラスが指定されています: "
                f"{invalid_classes}"
            )

            print(
                "使用可能なクラス: "
                f"{list(yolo.names.values())}"
            )

            return


    # ========================================================
    # YOLOのclass IDへ変換
    # ========================================================
    if TARGET_CLASSES is None:
        target_class_ids = None

    else:
        target_class_ids = [
            class_id
            for class_id, class_name in yolo.names.items()
            if class_name in TARGET_CLASSES
        ]

    print(
        f"対象class ID: {target_class_ids}"
    )


    # ========================================================
    # CRNN読み込み
    # ========================================================
    crnn = load_crnn_model()


    # ========================================================
    # 動画読み込み
    # ========================================================
    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        print(
            f"動画を開けませんでした: {VIDEO_PATH}"
        )
        return


    orig_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0


    frame_interval = max(
        1,
        int(round(
            orig_fps / TARGET_FPS
        ))
    )

    actual_fps = (
        orig_fps / frame_interval
    )


    video_name = os.path.splitext(
        os.path.basename(
            VIDEO_PATH
        )
    )[0]


    print("")
    print(
        f"処理開始: {VIDEO_PATH}"
    )

    print(
        f"入力FPS: {orig_fps:.3f}"
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
        "画像処理: カラーのまま高さ32リサイズのみ"
    )


    mode = (
        "a"
        if APPEND_LABEL_FILE
        else "w"
    )

    frame_idx = 0
    saved_count = 0


    # ========================================================
    # ラベルファイル
    # ========================================================
    with open(
        OUTPUT_LABEL_TXT,
        mode,
        encoding="utf-8"
    ) as out_f:

        while True:

            ret, frame = cap.read()

            if not ret:
                break


            # =================================================
            # 指定FPSのみ処理
            # =================================================
            if frame_idx % frame_interval == 0:

                # =============================================
                # YOLO推論
                #
                # classes= を指定することで、
                # YOLO推論段階から対象クラスだけに限定
                # =============================================
                if target_class_ids is None:

                    results = yolo(
                        frame,
                        conf=CONF_THRES,
                        verbose=False
                    )

                else:

                    results = yolo(
                        frame,
                        conf=CONF_THRES,
                        classes=target_class_ids,
                        verbose=False
                    )


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


                    # =========================================
                    # クラスごとにconfidence最大の1件
                    # =========================================
                    best_by_class = {}


                    for xyxy, cls_id, conf in zip(
                        boxes,
                        cls_ids,
                        confs
                    ):

                        cls_name = yolo.names[
                            int(cls_id)
                        ]


                        # =====================================
                        # 念のため再度対象クラスチェック
                        # =====================================
                        if (
                            TARGET_CLASSES is not None
                            and cls_name not in TARGET_CLASSES
                        ):
                            continue


                        if (
                            cls_id not in best_by_class
                            or conf >
                            best_by_class[
                                cls_id
                            ]["conf"]
                        ):
                            best_by_class[
                                cls_id
                            ] = {
                                "xyxy": xyxy,
                                "conf": conf
                            }


                    frame_records = []


                    # =========================================
                    # 選択されたクラスだけここへ入る
                    # =========================================
                    for cls_id, item in best_by_class.items():

                        cls_name = yolo.names[
                            int(cls_id)
                        ]


                        # =====================================
                        # 保存直前にもクラス確認
                        # =====================================
                        if (
                            TARGET_CLASSES is not None
                            and cls_name not in TARGET_CLASSES
                        ):
                            continue


                        x1, y1, x2, y2 = (
                            item["xyxy"]
                            .astype(int)
                        )

                        conf = item["conf"]


                        H, W = frame.shape[:2]


                        x1 = max(
                            0,
                            min(W - 1, x1)
                        )

                        x2 = max(
                            0,
                            min(W, x2)
                        )

                        y1 = max(
                            0,
                            min(H - 1, y1)
                        )

                        y2 = max(
                            0,
                            min(H, y2)
                        )


                        if (
                            x2 <= x1
                            or y2 <= y1
                        ):
                            continue


                        # =====================================
                        # 対象クラスの画像だけcrop
                        # =====================================
                        crop = frame[
                            y1:y2,
                            x1:x2
                        ]

                        if (
                            crop is None
                            or crop.size == 0
                        ):
                            continue


                        # =====================================
                        # カラーのまま高さ32
                        # =====================================
                        processed = preprocess_for_crnn(
                            crop
                        )

                        if processed is None:
                            continue


                        time_sec = (
                            frame_idx / orig_fps
                        )


                        # =====================================
                        # ファイル名
                        # =====================================
                        out_name = (
                            f"{video_name}_"
                            f"t{time_sec:06.2f}s_"
                            f"{cls_name}.jpg"
                        )

                        out_path = os.path.join(
                            OUTPUT_DIR,
                            out_name
                        )


                        # =====================================
                        # 選択クラスの画像だけ保存
                        # =====================================
                        success = cv2.imwrite(
                            out_path,
                            processed
                        )

                        if not success:
                            print(
                                f"画像保存失敗: {out_path}"
                            )
                            continue


                        # =====================================
                        # CRNN推論
                        # =====================================
                        text = predict_text(
                            crnn,
                            processed
                        )


                        frame_records.append({
                            "cls_name": cls_name,
                            "out_name": out_name,
                            "text": text,
                            "conf": conf,
                            "width": processed.shape[1],
                            "height": processed.shape[0]
                        })


                    # =========================================
                    # left → right
                    # =========================================
                    class_order = {
                        "left": 0,
                        "right": 1
                    }

                    frame_records.sort(
                        key=lambda x:
                        class_order.get(
                            x["cls_name"],
                            999
                        )
                    )


                    # =========================================
                    # 選択クラスだけlabels.txtへ保存
                    # =========================================
                    for rec in frame_records:

                        out_f.write(
                            f"{rec['out_name']} "
                            f"{rec['text']}\n"
                        )

                        saved_count += 1

                        print(
                            f"[{saved_count}] "
                            f"{rec['cls_name']} | "
                            f"{rec['out_name']} "
                            f"-> {rec['text']} "
                            f"(conf={rec['conf']:.3f}) "
                            f"size="
                            f"{rec['width']}x"
                            f"{rec['height']}"
                        )


            frame_idx += 1


    cap.release()


    print("")
    print("完了しました。")

    print(
        f"処理・保存対象クラス: {TARGET_CLASSES}"
    )

    print(
        f"保存枚数: {saved_count}"
    )

    print(
        f"画像保存先: {OUTPUT_DIR}"
    )

    print(
        f"ラベル出力先: {OUTPUT_LABEL_TXT}"
    )


if __name__ == "__main__":
    main()