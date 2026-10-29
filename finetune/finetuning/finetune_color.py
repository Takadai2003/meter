import os
import sys
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# ============================================================
# Import
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cnn_transformer_color.cnn_transformer import CRNN
from cnn_transformer_color import utils as color_utils

OCRDataset = color_utils.OCRDataset
collate_fn = color_utils.collate_fn
IDX2CHAR = color_utils.IDX2CHAR


IMG_DIR = "finetuning/output_readings_slip/train"
LABELS_TXT = "gemma4/gemma4_labels_slip.txt"

BASE_MODEL_PATH = "models/transformer_color/best-model_color_4.pth"
FINETUNED_SAVE_DIR = "models/finetuning_slip"
LATEST_MODEL_PATH = os.path.join(FINETUNED_SAVE_DIR, "latest.pth")

USE_LATEST_IF_EXISTS = False

BATCH = 64
EPOCHS = 30
LR = 5e-4
MIN_LR = 1e-4
WEIGHT_DECAY = 5e-3
CLIP_NORM = 5.0

VAL_RATIO = 0.1
SEED = 0

INPUT_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utility
# ============================================================

def calc_input_lengths(widths, max_t=None):
    lengths = torch.div(
        widths.to(torch.long),
        2,
        rounding_mode="floor"
    )
    lengths = torch.clamp(lengths, min=1)

    if max_t is not None:
        lengths = torch.clamp(lengths, max=max_t)

    return lengths


def unpack_batch(batch):
    if len(batch) == 5:
        imgs, targets_cat, tgt_lens, widths, paths = batch

    elif len(batch) == 4:
        imgs, targets_cat, tgt_lens, paths = batch
        widths = torch.full(
            (imgs.size(0),),
            imgs.size(-1),
            dtype=torch.long
        )

    else:
        raise ValueError(f"unexpected batch size: {len(batch)}")

    return imgs, targets_cat, tgt_lens, widths, paths


def ctc_decode(logits, input_lengths=None):
    best = logits.softmax(-1).argmax(-1).permute(1, 0)
    texts = []

    for i, seq in enumerate(best):
        if input_lengths is not None:
            seq = seq[:int(input_lengths[i].item())]

        out = []
        prev = None

        for idx in seq.tolist():
            if idx != 0 and idx != prev and idx < len(IDX2CHAR):
                out.append(IDX2CHAR[idx])
            prev = idx

        texts.append("".join(out))

    return texts


def targets_to_texts(targets_cat, tgt_lens):
    texts = []
    offset = 0

    for length in tgt_lens.tolist():
        seq = targets_cat[offset:offset + length].tolist()

        text = "".join(
            IDX2CHAR[idx]
            for idx in seq
            if idx != 0 and idx < len(IDX2CHAR)
        )

        texts.append(text)
        offset += length

    return texts


# ============================================================
# Model
# ============================================================

def choose_start_model_path():
    if USE_LATEST_IF_EXISTS and os.path.exists(LATEST_MODEL_PATH):
        return LATEST_MODEL_PATH

    # BASE_MODEL_PATH = None ならゼロから学習
    return BASE_MODEL_PATH


def load_model(nclass):
    model = CRNN(
        nclass=nclass,
        nc=INPUT_CHANNELS
    ).to(DEVICE)

    start_model_path = choose_start_model_path()

    if start_model_path is None:
        print("開始モデルなし: ゼロから学習します")
        print(f"CRNN input channels: {INPUT_CHANNELS}")

        return model, None

    if not os.path.exists(start_model_path):
        raise FileNotFoundError(
            f"開始モデルが見つかりません: {start_model_path}"
        )

    print(f"Loading model: {start_model_path}")

    state = torch.load(
        start_model_path,
        map_location=DEVICE
    )

    try:
        model.load_state_dict(state)

    except RuntimeError:
        print("")
        print("モデルの読み込みに失敗しました。")
        print("現在のCRNNは nc=3 です。")
        print("開始モデルも nc=3 で学習されている必要があります。")
        raise

    print(f"Loaded start model: {start_model_path}")
    print(f"CRNN input channels: {INPUT_CHANNELS}")

    return model, start_model_path


def save_models(model):
    os.makedirs(FINETUNED_SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamp_path = os.path.join(
        FINETUNED_SAVE_DIR,
        f"finetuned_{timestamp}.pth"
    )

    torch.save(model.state_dict(), timestamp_path)
    shutil.copy2(timestamp_path, LATEST_MODEL_PATH)

    print(f"saved timestamp model: {timestamp_path}")
    print(f"updated latest model : {LATEST_MODEL_PATH}")


# ============================================================
# Dataset check
# ============================================================

def check_dataset(dataset):
    img, target, width, path = dataset[0]

    print("")
    print("========== Dataset Check ==========")
    print(f"utils.py     : {color_utils.__file__}")
    print(f"sample image : {path}")
    print(f"tensor shape : {tuple(img.shape)}")
    print(f"width        : {width}")
    print(f"target length: {target.numel()}")

    if img.ndim != 3:
        raise RuntimeError(
            f"画像Tensorが[C,H,W]ではありません: {tuple(img.shape)}"
        )

    if img.shape[0] != INPUT_CHANNELS:
        raise RuntimeError(
            f"チャンネル数不一致: dataset={img.shape[0]}, "
            f"model={INPUT_CHANNELS}"
        )

    if img.shape[1] != 32:
        raise RuntimeError(
            f"画像高さが32ではありません: {img.shape[1]}"
        )

    print("Dataset: COLOR 3-channel OK")
    print("===================================")
    print("")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(FINETUNED_SAVE_DIR, exist_ok=True)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(
            f"画像フォルダが見つかりません: {IMG_DIR}"
        )

    if not os.path.exists(LABELS_TXT):
        raise FileNotFoundError(
            f"ラベルファイルが見つかりません: {LABELS_TXT}"
        )

    # Dataset
    ds = OCRDataset(IMG_DIR, LABELS_TXT)

    if len(ds) < 2:
        raise RuntimeError(f"データ数が少なすぎます: {len(ds)}")

    check_dataset(ds)

    # Train / Validation split
    n_val = max(1, int(VAL_RATIO * len(ds)))
    n_tr = len(ds) - n_val

    tr_ds, val_ds = random_split(
        ds,
        [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn
    )

    # Model
    nclass = len(IDX2CHAR)
    model, start_model_path = load_model(nclass)

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )

    best_acc = -1.0
    best_val_loss = float("inf")
    best_state = None

    print("")
    print("========== Training Settings ==========")
    print(f"DEVICE         : {DEVICE}")
    print(f"INPUT_CHANNELS : {INPUT_CHANNELS}")
    print(f"UTILS          : {color_utils.__file__}")
    print(f"IMG_DIR        : {IMG_DIR}")
    print(f"LABELS_TXT     : {LABELS_TXT}")
    print(f"DATASET_SIZE   : {len(ds)}")
    print(f"TRAIN_SIZE     : {n_tr}")
    print(f"VAL_SIZE       : {n_val}")
    print(f"START_MODEL    : {start_model_path or 'なし（ゼロから学習）'}")
    print(f"BATCH          : {BATCH}")
    print(f"EPOCHS         : {EPOCHS}")
    print(f"LR             : {LR}")
    print(f"MIN_LR         : {MIN_LR}")
    print(f"WEIGHT_DECAY   : {WEIGHT_DECAY}")
    print("=======================================")
    print("")

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        current_lr = optim.param_groups[0]["lr"]

        for batch in tr_loader:
            imgs, targets_cat, tgt_lens, widths, _ = unpack_batch(batch)

            imgs = imgs.to(
                DEVICE,
                non_blocking=True
            )

            targets_cat_device = targets_cat.to(
                DEVICE,
                non_blocking=True
            )

            inp_lens = calc_input_lengths(widths)

            seq_lens_device = inp_lens.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                imgs,
                seq_lens=seq_lens_device
            )

            T, _, _ = logits.size()
            inp_lens = torch.clamp(inp_lens, max=T)

            loss = criterion(
                logits.log_softmax(-1),
                targets_cat_device,
                inp_lens,
                tgt_lens
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CLIP_NORM
            )

            optim.step()
            total_train_loss += loss.item()

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, targets_cat, tgt_lens, widths, _ = unpack_batch(batch)

                imgs = imgs.to(
                    DEVICE,
                    non_blocking=True
                )

                targets_cat_device = targets_cat.to(
                    DEVICE,
                    non_blocking=True
                )

                inp_lens = calc_input_lengths(widths)

                seq_lens_device = inp_lens.to(
                    DEVICE,
                    non_blocking=True
                )

                logits = model(
                    imgs,
                    seq_lens=seq_lens_device
                )

                T, _, _ = logits.size()
                inp_lens = torch.clamp(inp_lens, max=T)

                loss = criterion(
                    logits.log_softmax(-1),
                    targets_cat_device,
                    inp_lens,
                    tgt_lens
                )

                total_val_loss += loss.item()

                pred_texts = ctc_decode(
                    logits,
                    inp_lens
                )

                gt_texts = targets_to_texts(
                    targets_cat,
                    tgt_lens
                )

                for pred, gt in zip(pred_texts, gt_texts):
                    if pred == gt:
                        val_correct += 1

                    val_total += 1

        train_loss = total_train_loss / max(1, len(tr_loader))
        val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        print(
            f"[{epoch:02d}/{EPOCHS}] "
            f"lr={current_lr:.8f} "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"acc={val_acc:.4f} "
            f"({val_correct}/{val_total})"
        )

        if (
            val_acc > best_acc
            or (
                val_acc == best_acc
                and val_loss < best_val_loss
            )
        ):
            best_acc = val_acc
            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            print(
                f"best updated: "
                f"acc={best_acc:.4f}, "
                f"val_loss={best_val_loss:.4f}"
            )

        scheduler.step()

    # Best model
    if best_state is not None:
        model.load_state_dict(best_state)

    save_models(model)

    print("")
    print("finetuning finished")
    print(f"best_acc     : {best_acc:.4f}")
    print(f"best_val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()