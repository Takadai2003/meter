import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from cnn_transformer import CRNN
from utils import OCRDataset, collate_fn, IDX2CHAR

IMG_DIR = "learning/resized"
LABELS_TXT = "learning/labels_datasets.txt"
SAVE_DIR = "learning/models/hugging_face/Transformer"
BATCH = 64
EPOCHS = 50
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-5
CLIP_NORM = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calc_input_lengths(widths, max_t=None):
    widths = widths.to(torch.long)
    lengths = torch.div(widths, 2, rounding_mode="floor")
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
            size=(imgs.size(0),),
            fill_value=imgs.size(-1),
            dtype=torch.long
        )
    else:
        raise ValueError(f"unexpected batch size: {len(batch)}")

    return imgs, targets_cat, tgt_lens, widths, paths

def ctc_decode(logits, input_lengths=None):
    probs = logits.softmax(-1)
    best = probs.argmax(-1)
    best = best.permute(1, 0)
    texts = []

    for i, seq in enumerate(best):
        if input_lengths is not None:
            valid_len = int(input_lengths[i].item())
            seq = seq[:valid_len]

        out, prev = [], None

        for idx in seq.tolist():
            if idx != 0 and idx != prev:
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

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    ds = OCRDataset(IMG_DIR, LABELS_TXT)

    n_val = max(1, int(0.1 * len(ds)))
    n_tr = len(ds) - n_val

    tr_ds, val_ds = random_split(
        ds,
        [n_tr, n_val],
        generator=torch.Generator().manual_seed(0)
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

    nclass = len(IDX2CHAR)
    model = CRNN(nclass=nclass, nc=1).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

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
    best_val_loss = 1e9

    print(f"DEVICE: {DEVICE}")
    print(f"BATCH: {BATCH}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"MIN_LR: {MIN_LR}")
    print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        current_lr = optim.param_groups[0]["lr"]

        for batch in tr_loader:
            imgs, targets_cat, tgt_lens, widths, _ = unpack_batch(batch)

            imgs = imgs.to(DEVICE, non_blocking=True)
            targets_cat_device = targets_cat.to(DEVICE, non_blocking=True)

            inp_lens = calc_input_lengths(widths)
            seq_lens_device = inp_lens.to(DEVICE, non_blocking=True)

            logits = model(imgs, seq_lens=seq_lens_device)

            T, B, C = logits.size()
            inp_lens = torch.clamp(inp_lens, max=T)

            logp = logits.log_softmax(-1)

            loss = criterion(
                logp,
                targets_cat_device,
                inp_lens,
                tgt_lens
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optim.step()

            tot_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, targets_cat, tgt_lens, widths, _ = unpack_batch(batch)

                imgs = imgs.to(DEVICE, non_blocking=True)
                targets_cat_device = targets_cat.to(DEVICE, non_blocking=True)

                inp_lens = calc_input_lengths(widths)
                seq_lens_device = inp_lens.to(DEVICE, non_blocking=True)

                logits = model(imgs, seq_lens=seq_lens_device)

                T, B, C = logits.size()
                inp_lens = torch.clamp(inp_lens, max=T)

                logp = logits.log_softmax(-1)

                loss = criterion(
                    logp,
                    targets_cat_device,
                    inp_lens,
                    tgt_lens
                )

                val_loss += loss.item()

                pred_texts = ctc_decode(logits, inp_lens)
                gt_texts = targets_to_texts(targets_cat, tgt_lens)

                for pred, gt in zip(pred_texts, gt_texts):
                    if pred == gt:
                        val_correct += 1
                    val_total += 1

        tr_loss = tot_loss / max(1, len(tr_loader))
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        print(
            f"[{epoch}/{EPOCHS}] "
            f"lr {current_lr:.8f}  "
            f"train {tr_loss:.4f}  "
            f"val {val_loss:.4f}  "
            f"acc {val_acc:.4f} "
            f"({val_correct}/{val_total})"
        )

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_val_loss):
            best_acc = val_acc
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "best_datasets.pth")
            torch.save(model.state_dict(), save_path)
            print(f"saved {save_path} acc={best_acc:.4f} val_loss={best_val_loss:.4f}")

        scheduler.step()

    last_path = os.path.join(SAVE_DIR, "last_datasets.pth")
    torch.save(model.state_dict(), last_path)
    print(f"saved {last_path}")

if __name__ == "__main__":
    main()