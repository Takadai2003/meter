import os
import cv2


# ============================================================
# 設定
# ============================================================

VIDEO_PATH = "/mnt/c/Users/大西研究室/Desktop/machine2_slip.mp4"
OUTPUT_DIR = "datasets3_yolo/images/train"

# 処理・保存したいFPS
TARGET_FPS = 0.014

IMAGE_EXT = ".jpg"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"動画を開けませんでした: {VIDEO_PATH}")
        return

    # 元動画情報
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0

    if TARGET_FPS <= 0:
        raise ValueError("TARGET_FPS は0より大きくしてください")

    if TARGET_FPS > orig_fps:
        print(
            f"TARGET_FPS={TARGET_FPS} は入力FPS={orig_fps:.3f}を超えています。"
        )
        print("入力FPSと同じ間隔で処理します。")

    # 何フレームおきに処理するか
    frame_interval = max(
        1,
        int(round(orig_fps / TARGET_FPS))
    )

    # 実際に処理されるFPS
    actual_fps = orig_fps / frame_interval

    video_name = os.path.splitext(
        os.path.basename(VIDEO_PATH)
    )[0]

    print(f"動画: {VIDEO_PATH}")
    print(f"入力FPS: {orig_fps:.3f}")
    print(f"指定TARGET_FPS: {TARGET_FPS}")
    print(f"実際の処理FPS: {actual_fps:.3f}")
    print(f"frame_interval: {frame_interval}")
    print(f"総フレーム数: {total_frames}")
    print("")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 指定FPSに相当するフレームだけ処理
        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / orig_fps

            output_name = (
                f"{video_name}_"
                f"frame_{frame_idx:06d}_"
                f"t{time_sec:07.2f}s"
                f"{IMAGE_EXT}"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            success = cv2.imwrite(
                output_path,
                frame
            )

            if success:
                saved_count += 1

                print(
                    f"[{saved_count}] "
                    f"frame={frame_idx} "
                    f"time={time_sec:.2f}s "
                    f"-> {output_name}"
                )
            else:
                print(
                    f"保存失敗: {output_path}"
                )

        frame_idx += 1

    cap.release()

    print("")
    print("完了しました。")
    print(f"保存枚数: {saved_count}")
    print(f"保存先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()