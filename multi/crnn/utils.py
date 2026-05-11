import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

ALPHABET = "0123456789.-"
IDX2CHAR = ["-"] + list(ALPHABET)
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}

IMG_HEIGHT = 32
PAD_VALUE = -1.0

def text_to_indices(s: str):
    return [CHAR2IDX[c] for c in s if c in CHAR2IDX]

def indices_to_text(idxs):
    out, prev = [], None

    for i in idxs:
        if i != prev and i != 0:
            out.append(IDX2CHAR[i])
        prev = i

    return "".join(out)

def resize_keep_height(img, target_h=32):
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("invalid image size")

    scale = target_h / h
    new_w = max(1, int(round(w * scale)))

    interpolation = cv2.INTER_NEAREST

    img = cv2.resize(
        img,
        (new_w, target_h),
        interpolation=interpolation
    )

    return img

class OCRDataset(Dataset):
    def __init__(self, img_dir, labels_txt):
        self.img_dir = img_dir
        self.samples = []

        with open(labels_txt, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                name, *lab = line.strip().split()
                img_path = os.path.join(img_dir, name)
                label = " ".join(lab)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(path)

        if img.shape[0] != IMG_HEIGHT:
            img = resize_keep_height(img, target_h=IMG_HEIGHT)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        img = torch.from_numpy(img).unsqueeze(0)

        width = img.shape[-1]
        target = torch.tensor(text_to_indices(text), dtype=torch.long)

        return img, target, width, path

def collate_fn(batch):
    imgs, targets, widths, paths = zip(*batch)

    max_w = max(img.shape[-1] for img in imgs)

    padded_imgs = []

    for img in imgs:
        c, h, w = img.shape
        pad_w = max_w - w

        if pad_w > 0:
            pad = torch.full(
                (c, h, pad_w),
                PAD_VALUE,
                dtype=img.dtype
            )
            img = torch.cat([img, pad], dim=2)

        padded_imgs.append(img)

    imgs = torch.stack(padded_imgs, 0)

    lengths = torch.tensor(
        [t.numel() for t in targets],
        dtype=torch.long
    )

    widths = torch.tensor(
        widths,
        dtype=torch.long
    )

    if len(targets) > 0:
        targets_cat = torch.cat(targets)
    else:
        targets_cat = torch.tensor([], dtype=torch.long)

    return imgs, targets_cat, lengths, widths, paths