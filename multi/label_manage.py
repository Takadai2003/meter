import os
from pathlib import Path

# --- パスの設定 ---
# 画像フォルダのパス
image_folder_path = "crop"

# 元のラベルファイルのパス (※crnnフォルダ内にあると仮定しています。適宜修正してください)
labels_file_path = "crnn/labels.txt"

# クリーニング後の新しいラベルファイルの保存先
output_labels_path = "crnn/labels_cleaned.txt"


def clean_labels(image_folder, labels_file, output_file):
    image_dir = Path(image_folder)
    labels_path = Path(labels_file)

    # ラベルファイルが存在するか確認
    if not labels_path.exists():
        print(f"エラー: ラベルファイルが見つかりません。\nパス: {labels_file}")
        return

    # 元のラベルファイルを読み込む
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_lines = []
    removed_count = 0

    print("ラベルの照合を開始します...")
    print("-" * 30)

    for line in lines:
        # 空行をスキップ
        if not line.strip():
            continue
            
        # 行を空白で分割し、最初の要素（ファイル名）を取得
        # 例: "WIN_..._0062_left.jpg 2.4" -> parts[0]は "WIN_..._0062_left.jpg"
        parts = line.strip().split()
        filename = parts[0]
        
        # 該当する画像ファイルがフォルダに存在するかチェック
        image_file_path = image_dir / filename
        
        if image_file_path.exists():
            # 存在する場合はリストに追加（保持）
            valid_lines.append(line)
        else:
            # 存在しない場合はリストに追加しない（削除）
            print(f"[行削除] 画像が見つかりません: {filename}")
            removed_count += 1

    # 新しいテキストファイルに書き出す
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)

    print("-" * 30)
    print("処理が完了しました！")
    print(f"保持したラベル数: {len(valid_lines)}")
    print(f"削除したラベル数: {removed_count}")
    print(f"\nクリーンアップされたファイルを以下に保存しました:\n{output_file}")


if __name__ == "__main__":
    clean_labels(image_folder_path, labels_file_path, output_labels_path)