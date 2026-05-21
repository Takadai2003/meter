import os
from pathlib import Path

folder_path = "crop"

def process_files(target_folder):
    target_dir = Path(target_folder)

    # フォルダが存在するか確認
    if not target_dir.exists():
        print(f"エラー: 指定されたフォルダが見つかりません。\nパス: {target_folder}")
        return

    files = sorted([f for f in target_dir.iterdir() if f.is_file()])
    
    if not files:
        print("フォルダ内にファイルがありません。")
        return

    deleted_count = 0
    keep_count = 0

    print(f"処理を開始します: {target_dir.absolute()}")
    print("-" * 30)

    # 「2つおきに2つ消す」ロジック
    for i, file_path in enumerate(files):
        if (i % 4) >= 2:
            try:
                # 実際に削除を実行
                file_path.unlink()
                print(f"[削除] {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"[エラー] {file_path.name} を削除できませんでした: {e}")
        else:
            print(f"[保持] {file_path.name}")
            keep_count += 1

    print("-" * 30)
    print(f"処理が完了しました！")
    print(f"残したファイル数: {keep_count}")
    print(f"削除したファイル数: {deleted_count}")

if __name__ == "__main__":
    process_files(folder_path)