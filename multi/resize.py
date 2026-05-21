import cv2
import os
import numpy as np

input_folder = "crnn/test_cropped"
output_folder = "crnn/test_resized"

target_size = (100, 32)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    #アスペクト比を維持したままのスケールを算出
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    #計算したスケールでリサイズ
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    #目標サイズの余白に黒背景追加
    canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 0
    #黒背景の中央に画像配置
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, canvas)
    print(f"保存: {out_path}")