import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- 設定項目 ---
VIDEO_PATH = "/mnt/g/共有ドライブ/量子情報デバイス研究室/修士2026/髙橋大騎/machine2の動画/machine2_shake.mp4"
OUTPUT_DIR = 'cropped'
MODEL_PATH = 'models/text-best.pt'

TARGET_FPS = 0.1
TARGET_SIZE = (100, 32)  # (W, H)

# YOLO設定
YOLO_CONF_THRES = 0.10

# 前処理パラメータ
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (4, 4)
THRESH_VALUE = 105

# ノイズ除去設定
REMOVE_SMALL_WHITE_NOISE = True
MAX_NOISE_AREA = 100  # ノイズ除去面積

# ========= 射影変換設定 =========
USE_PERSPECTIVE_CORRECTION = True

# 四角形候補の最大数
MAX_RECT_CANDIDATES = 30

# 四角形を外側に広げる倍率
# 1.00なら検出された四角形そのまま
# 数字が端で切れる場合だけ 1.03〜1.08 程度に上げる
WARP_MARGIN_SCALE = 1.00

# 四角形候補の面積条件
MIN_RECT_AREA_RATIO = 0.02
MAX_RECT_AREA_RATIO = 0.60

# 四角形候補の縦横比条件
# 青パネルがほぼ正方形なので、かなり広すぎる候補は除外する
MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.80

# Cannyエッジ検出パラメータ
CANNY_TH1 = 30
CANNY_TH2 = 100

# 四角形近似の強さ
# 小さいほど輪郭を細かく残す
APPROX_EPS_RATIO = 0.015

# 射影変換後サイズ
# 青パネルがほぼ正方形なのでまずは 640x640
# YOLOが検出しにくい場合は (640, 480) も試す
WARP_OUTPUT_SIZE = (640, 640)  # (W, H)

# 候補でYOLO検出できなかった場合に元画像で処理するか
# 原因確認中は False 推奨。Trueにすると元画像が warped として保存される場合がある。
FALLBACK_TO_ORIGINAL_IF_NO_WARP_HIT = False

# デバッグ画像を保存するか
SAVE_DEBUG_QUAD = True

# --- 出力フォルダ作成 ---
WARPED_DIR = os.path.join(OUTPUT_DIR, "warped")
YOLO_CROP_DIR = os.path.join(OUTPUT_DIR, "yolo_crop")
RESIZED_DIR = os.path.join(OUTPUT_DIR, "resized")
DEBUG_QUAD_DIR = os.path.join(OUTPUT_DIR, "debug_quad")

os.makedirs(WARPED_DIR, exist_ok=True)
os.makedirs(YOLO_CROP_DIR, exist_ok=True)
os.makedirs(RESIZED_DIR, exist_ok=True)
os.makedirs(DEBUG_QUAD_DIR, exist_ok=True)

# --- モデルロード ---
model = YOLO(MODEL_PATH)


def remove_small_white_components(binary_img, max_area=80):
    cleaned = binary_img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    for label in range(1, num_labels):  # 0 は背景
        _, _, _, _, area = stats[label]

        if area <= max_area:
            cleaned[labels == label] = 0

    return cleaned


def order_points(pts):
    """
    4点を 左上, 右上, 右下, 左下 の順に並べ替える。
    """
    pts = np.array(pts, dtype="float32")

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype="float32")


def has_duplicate_points(pts, min_dist=5.0):
    """
    4点の中に近すぎる点がないか確認する。
    """
    pts = np.array(pts, dtype="float32")

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                return True

    return False


def quad_area(pts):
    pts = np.array(pts, dtype="float32")
    return abs(cv2.contourArea(pts))


def quad_aspect_ratio(pts):
    """
    四角形の概算 width / height を返す。
    """
    pts = order_points(pts)
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    width = max(width_top, width_bottom)
    height = max(height_right, height_left)

    if height <= 1e-6:
        return None

    return width / height


def is_convex_quad(pts):
    """
    四角形が凸形状か確認する。
    """
    pts = order_points(pts).astype(np.int32)
    pts = pts.reshape(-1, 1, 2)
    return cv2.isContourConvex(pts)


def is_valid_quad(pts, frame_shape):
    """
    面積・縦横比・凸性から四角形候補として妥当か判定する。
    """
    h, w = frame_shape[:2]
    frame_area = h * w

    pts = order_points(pts)

    if has_duplicate_points(pts):
        return False

    if not is_convex_quad(pts):
        return False

    area = quad_area(pts)

    if area < frame_area * MIN_RECT_AREA_RATIO:
        return False

    if area > frame_area * MAX_RECT_AREA_RATIO:
        return False

    aspect = quad_aspect_ratio(pts)
    if aspect is None:
        return False

    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False

    return True


def bbox_from_quad(pts):
    pts = np.array(pts, dtype="float32")
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    return x, y, x + w, y + h


def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def deduplicate_quads(candidates, iou_thres=0.80):
    """
    似た四角形候補を削除する。
    """
    kept = []

    for cand in candidates:
        box = bbox_from_quad(cand["pts"])

        duplicate = False
        for kept_cand in kept:
            kept_box = bbox_from_quad(kept_cand["pts"])
            if bbox_iou(box, kept_box) > iou_thres:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


def contour_to_extreme_quad(cnt):
    """
    輪郭点群から、左上・右上・右下・左下に相当する4点を推定する。

    重要:
    minAreaRect のような外接長方形ではなく、
    実際の輪郭上の点を使う。
    """
    pts = cnt.reshape(-1, 2).astype("float32")

    if len(pts) < 4:
        return None

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    quad = np.array([tl, tr, br, bl], dtype="float32")
    quad = order_points(quad)

    if has_duplicate_points(quad):
        return None

    return quad


def detect_rectangle_candidates(frame):
    """
    色に依存せず、エッジと輪郭から四角形候補を複数検出する。

    今回の修正点:
    minAreaRect は使わない。
    理由:
        minAreaRect は輪郭を囲む外接回転矩形なので、
        斜めに写った表示部の台形形状を保持できない。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ノイズ低減
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # エッジ検出
    edges = cv2.Canny(blur, CANNY_TH1, CANNY_TH2)

    # エッジをつなげる
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area <= 0:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        # 1) まず通常の四角形近似を試す
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_RATIO * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            pts = order_points(pts)

            if is_valid_quad(pts, frame.shape):
                candidates.append({
                    "pts": pts,
                    "area": quad_area(pts),
                    "method": "approx4"
                })

        # 2) 丸角などで4点にならない場合、輪郭上の極点から4隅を推定する
        if len(approx) >= 4:
            pts = contour_to_extreme_quad(cnt)

            if pts is not None and is_valid_quad(pts, frame.shape):
                candidates.append({
                    "pts": pts,
                    "area": quad_area(pts),
                    "method": "extreme_quad"
                })

    # 面積が大きい順に並べる
    candidates = sorted(candidates, key=lambda x: x["area"], reverse=True)

    # 重複候補を削除
    candidates = deduplicate_quads(candidates, iou_thres=0.80)

    # 上位候補だけ返す
    return candidates[:MAX_RECT_CANDIDATES]


def expand_quad(pts, scale, frame_shape):
    """
    四角形を中心から外側に広げる。
    """
    h, w = frame_shape[:2]

    pts = order_points(pts)
    center = np.mean(pts, axis=0)

    expanded = (pts - center) * scale + center

    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    return order_points(expanded)


def get_warp_size_from_quad(pts):
    """
    検出四角形の辺長から射影変換後サイズを決める。
    WARP_OUTPUT_SIZE = None のときだけ使う。
    """
    pts = order_points(pts)
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    width = int(max(width_top, width_bottom))
    height = int(max(height_right, height_left))

    width = max(width, 32)
    height = max(height, 32)

    return width, height


def warp_by_quad(frame, pts):
    """
    四角形領域を射影変換して正面化する。
    """
    pts = order_points(pts)

    if WARP_OUTPUT_SIZE is None:
        width, height = get_warp_size_from_quad(pts)
    else:
        width, height = WARP_OUTPUT_SIZE

    dst_pts = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst_pts)
    warped = cv2.warpPerspective(frame, M, (width, height))

    return warped


def draw_quad_debug(frame, pts, out_path, label="selected"):
    """
    採用された四角形を元画像上に描画して保存する。
    """
    debug = frame.copy()
    pts = order_points(pts).astype(int)

    # 四角形を赤線で描画
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

    # 頂点を描画
    names = ["TL", "TR", "BR", "BL"]
    for p, name in zip(pts, names):
        x, y = p
        cv2.circle(debug, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            debug,
            name,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        debug,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.imwrite(out_path, debug)


def yolo_score(results):
    """
    YOLO検出結果をスコア化する。
    left/right の両方がある候補を最優先する。
    """
    det_count = 0
    conf_sum = 0.0
    class_names = set()

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        det_count += len(confs)
        conf_sum += float(np.sum(confs))

        for cls_id in cls_ids:
            class_names.add(str(model.names[int(cls_id)]))

    has_left = "left" in class_names
    has_right = "right" in class_names

    if has_left and has_right:
        class_bonus = 1000.0
    elif has_left or has_right:
        class_bonus = 200.0
    else:
        class_bonus = 0.0

    score = class_bonus + det_count * 10.0 + conf_sum

    return score, det_count, conf_sum, class_names


def choose_best_perspective_frame(frame):
    """
    四角形候補を複数作成し、それぞれにYOLOをかける。
    YOLO検出結果が最も良い射影変換画像を採用する。
    """
    candidates = detect_rectangle_candidates(frame)

    best = None

    for cand_idx, cand in enumerate(candidates):
        pts = cand["pts"]

        # 余白付きで射影変換
        pts_expanded = expand_quad(
            pts,
            scale=WARP_MARGIN_SCALE,
            frame_shape=frame.shape
        )

        warped = warp_by_quad(frame, pts_expanded)

        if warped is None or warped.size == 0:
            continue

        results = model(warped, conf=YOLO_CONF_THRES, verbose=False)
        score, det_count, conf_sum, class_names = yolo_score(results)

        # YOLOが何も検出しない候補は不採用
        if det_count == 0:
            continue

        if best is None or score > best["score"]:
            best = {
                "warped": warped,
                "results": results,
                "score": score,
                "det_count": det_count,
                "conf_sum": conf_sum,
                "class_names": class_names,
                "candidate_index": cand_idx,
                "method": cand["method"],
                "pts": pts,
                "pts_expanded": pts_expanded,
                "raw_area": cand["area"]
            }

    if best is not None:
        return best

    # 射影変換候補でYOLO検出できなかった場合、元画像にフォールバックするか
    if FALLBACK_TO_ORIGINAL_IF_NO_WARP_HIT:
        results = model(frame, conf=YOLO_CONF_THRES, verbose=False)
        score, det_count, conf_sum, class_names = yolo_score(results)

        if det_count > 0:
            return {
                "warped": frame,
                "results": results,
                "score": score,
                "det_count": det_count,
                "conf_sum": conf_sum,
                "class_names": class_names,
                "candidate_index": -1,
                "method": "original",
                "pts": None,
                "pts_expanded": None,
                "raw_area": None
            }

    return None


# --- 動画読み込み ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"動画を開けませんでした: {VIDEO_PATH}")
    raise SystemExit

# --- 動画情報 ---
orig_fps = cap.get(cv2.CAP_PROP_FPS)
if not orig_fps or orig_fps <= 0:
    orig_fps = 30.0

frame_interval = max(1, int(round(orig_fps / TARGET_FPS)))
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

print(f"処理開始: {VIDEO_PATH}")
print(f"入力FPS: {orig_fps:.2f} / 抽出FPS: {TARGET_FPS} / interval={frame_interval}")

# --- CLAHE生成 ---
clahe = None
if USE_CLAHE:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )

frame_idx = 0
saved_count = 0
warped_saved_count = 0
crop_saved_count = 0
debug_saved_count = 0
warp_failed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 指定FPSでフレーム抽出
    if frame_idx % frame_interval == 0:

        # ========= 0) YOLO前に射影変換候補を探索 =========
        if USE_PERSPECTIVE_CORRECTION:
            selected = choose_best_perspective_frame(frame)

            if selected is None:
                warp_failed_count += 1
                print(f"射影変換候補なし / YOLO検出なし: frame={frame_idx}")
                frame_idx += 1
                continue

            input_frame = selected["warped"]
            results = selected["results"]

            # 射影変換後画像を保存
            warped_name = (
                f"{video_name}_f{frame_idx:06d}_"
                f"{selected['method']}_cand{selected['candidate_index']}_warped.jpg"
            )
            warped_path = os.path.join(WARPED_DIR, warped_name)
            cv2.imwrite(warped_path, input_frame)
            warped_saved_count += 1

            # 採用された四角形を元画像上に描画して保存
            if SAVE_DEBUG_QUAD and selected["pts_expanded"] is not None:
                debug_name = (
                    f"{video_name}_f{frame_idx:06d}_"
                    f"{selected['method']}_cand{selected['candidate_index']}_quad.jpg"
                )
                debug_path = os.path.join(DEBUG_QUAD_DIR, debug_name)

                draw_quad_debug(
                    frame,
                    selected["pts_expanded"],
                    debug_path,
                    label=(
                        f"{selected['method']} "
                        f"cand={selected['candidate_index']} "
                        f"score={selected['score']:.1f}"
                    )
                )
                debug_saved_count += 1

            print(
                f"採用: frame={frame_idx}, method={selected['method']}, "
                f"candidate={selected['candidate_index']}, "
                f"det={selected['det_count']}, score={selected['score']:.3f}, "
                f"classes={sorted(list(selected['class_names']))}"
            )

        else:
            input_frame = frame
            results = model(input_frame, conf=YOLO_CONF_THRES, verbose=False)

        # ========= 1) 採用画像に対するYOLO結果からcrop =========
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for det_idx, (xyxy, cls_id) in enumerate(zip(boxes, cls_ids)):
                x1, y1, x2, y2 = xyxy

                # 枠外クリップ
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(input_frame.shape[1], x2)
                y2 = min(input_frame.shape[0], y2)

                # 無効領域はスキップ
                if x2 <= x1 or y2 <= y1:
                    continue

                # 2) YOLO crop
                crop = input_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # クラス名取得
                cls_name = model.names[int(cls_id)]

                # YOLO切り出し後画像を保存
                crop_name = f"{video_name}_f{frame_idx:06d}_d{det_idx}_{cls_name}_crop.jpg"
                crop_path = os.path.join(YOLO_CROP_DIR, crop_name)
                cv2.imwrite(crop_path, crop)
                crop_saved_count += 1

                # 3) Gチャネル抽出
                gray = crop[:, :, 1]

                # 4) CLAHE
                if clahe is not None:
                    gray = clahe.apply(gray)

                # 5) binary
                _, binary = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY)

                # 6) 小さい白ノイズ除去
                if REMOVE_SMALL_WHITE_NOISE:
                    binary = remove_small_white_components(
                        binary,
                        max_area=MAX_NOISE_AREA
                    )

                # 7) 100x32
                out_img = cv2.resize(
                    binary,
                    TARGET_SIZE,
                    interpolation=cv2.INTER_AREA
                )

                # 最終リサイズ後画像を保存
                out_name = f"{video_name}_f{frame_idx:06d}_d{det_idx}_{cls_name}.jpg"
                out_path = os.path.join(RESIZED_DIR, out_name)
                cv2.imwrite(out_path, out_img)

                saved_count += 1
                print(f"保存: {out_path}")

        if frame_idx % (frame_interval * 10) == 0:
            print(
                f"--- 処理中: {frame_idx}フレーム目 "
                f"(warped保存: {warped_saved_count}枚 / "
                f"debug保存: {debug_saved_count}枚 / "
                f"crop保存: {crop_saved_count}枚 / "
                f"resize保存: {saved_count}枚 / "
                f"失敗: {warp_failed_count}回) ---"
            )

    frame_idx += 1

cap.release()

print("完了！")
print(f"射影変換後画像: {warped_saved_count} 枚")
print(f"debug_quad画像: {debug_saved_count} 枚")
print(f"YOLO切り出し画像: {crop_saved_count} 枚")
print(f"最終リサイズ画像: {saved_count} 枚")
print(f"射影変換候補なし / YOLO検出なし: {warp_failed_count} 回")