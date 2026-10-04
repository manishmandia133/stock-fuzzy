"""Training loop with regime-aware batching + 7 variants (PLAN T5 + T6).

Batching: train windows pre-sorted by end_date, shuffle=False at sample level;
each epoch permutes the ORDER of batches, not samples.
Batch volatility = mean vol_rank of windows in the batch.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as cfg
import fuzzy as fz
from model import DirectionLSTM

VARIANTS = ["fuzzy"]


def _batch_index_list(n, batch_size):
    n_batches = n // batch_size  # drop_last=True
    return [np.arange(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def _variant_lr(variant, batch_vol, rng, cos_lr=None):
    if variant == "fuzzy":
        return fz.fuzzy_lr(batch_vol)
    raise ValueError(variant)


def train_variant(X_train, y_train, vr_train, X_val, y_val,
                  variant, seed, epochs=cfg.EPOCHS,
                  batch_size=cfg.BATCH_SIZE, logdir="logs"):
    """Train one variant/seed. Returns (model, history, batch_log_df, info)."""
    cfg.set_seed(seed)
    os.makedirs(logdir, exist_ok=True)
    device = cfg.DEVICE

    model = DirectionLSTM(n_features=X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.FIXED_LR)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if variant == "cosine" else None)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    batches = _batch_index_list(len(Xt), batch_size)
    rng = np.random.default_rng(seed)

    # T5 gate quantity: std of batch vol (fixed batching, before permutation)
    bvols = np.array([vr_train[b].mean() for b in batches])
    vol_std = float(bvols.std())

    rows = []
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(batches))  # permute batch ORDER only
        cos_lr = scheduler.get_last_lr()[0] if scheduler else None
        model.train()
        run_loss, n_seen = 0.0, 0
        for bi in order:
            idx = batches[bi]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            bvol = float(vr_train[idx].mean())
            lr = _variant_lr(variant, bvol, rng, cos_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            run_loss += loss.item() * len(idx)
            n_seen += len(idx)
            rows.append({"epoch": epoch, "batch": int(bi),
                         "vol": bvol, "lr": lr, "loss": loss.item()})

        if scheduler:
            scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            vloss = nn.BCELoss()(pv, yv).item()
            vacc = (((pv > 0.5).float() == yv).sum().item() / max(len(yv), 1))
        history["train_loss"].append(run_loss / max(n_seen, 1))
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)

        if vloss < best_loss - 1e-9:
            best_loss, best_state, best_epoch, stale = vloss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= cfg.PATIENCE:
                print(f"[{variant} s{seed}] early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(best_state)
    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(logdir, f"{variant}_s{seed}.csv"), index=False)

    # T6 gate (fuzzy variant): corr(vol, lr) < -0.7
    corr = float(log_df["vol"].corr(log_df["lr"])) if log_df["lr"].std() > 0 else float("nan")
    info = {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "batch_vol_std": vol_std, "vol_lr_corr": corr}
    print(f"[{variant} s{seed}] best_epoch={best_epoch} val_loss={best_loss:.4f} "
          f"vol_std={vol_std:.3f} corr(vol,lr)={corr:.3f}")
    return model, history, log_df, info
