import os
import shutil
from pathlib import Path


INPUT_DIR = "gemma4/gemma4_images"

LEFT_DIR = os.path.join(INPUT_DIR, "left")
RIGHT_DIR = os.path.join(INPUT_DIR, "right")

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# True: コピーする
# False: 移動する
COPY_MODE = True

# 同名ファイルが出力先にある場合に上書きするか
OVERWRITE = True


def get_class_from_filename(path: Path):
    stem = path.stem

    if stem.endswith("_left"):
        return "left"

    if stem.endswith("_right"):
        return "right"

    return None


def copy_or_move(src_path: Path, dst_path: Path):
    if dst_path.exists():
        if OVERWRITE:
            dst_path.unlink()
        else:
            print(f"既に存在するためスキップ: {dst_path}")
            return False

    if COPY_MODE:
        shutil.copy2(src_path, dst_path)
    else:
        shutil.move(str(src_path), str(dst_path))

    return True


def main():
    input_path = Path(INPUT_DIR)

    if not input_path.exists():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {INPUT_DIR}")

    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    left_count = 0
    right_count = 0
    skip_count = 0

    for path in sorted(input_path.iterdir()):
        # left/right フォルダ自体は無視
        if path.is_dir():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        class_name = get_class_from_filename(path)

        if class_name == "left":
            dst_path = Path(LEFT_DIR) / path.name
            saved = copy_or_move(path, dst_path)

            if saved:
                left_count += 1
                print(f"left に保存: {dst_path}")

        elif class_name == "right":
            dst_path = Path(RIGHT_DIR) / path.name
            saved = copy_or_move(path, dst_path)

            if saved:
                right_count += 1
                print(f"right に保存: {dst_path}")

        else:
            skip_count += 1
            print(f"クラス判定できないためスキップ: {path.name}")

    print("")
    print("完了")
    print(f"left 画像数 : {left_count}")
    print(f"right画像数 : {right_count}")
    print(f"skip数      : {skip_count}")
    print(f"元画像を残す: {COPY_MODE}")


if __name__ == "__main__":
    main()