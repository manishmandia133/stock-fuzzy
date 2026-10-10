"""Training loop with regime-aware batching + 5 variants.

Batching: train windows pre-sorted by end_date; each epoch permutes the ORDER
of contiguous batches (never shuffles samples across batches). The final
partial batch is INCLUDED (no silent drop_last).
Variants:
  fuzzy              - Mamdani fuzzy LR from batch mean vol_rank
  fixed_1e-3         - constant 1e-3
  fixed_mean_fuzzy   - constant == mean of fuzzy curve (same average budget)
  fuzzy_inverted     - reversed curve (low vol -> low LR; control)
  fuzzy_shuffled_vol - LR drawn from the same empirical distribution
                       (random train vol_rank -> fuzzy LR), decoupled from
                       the batch's own volatility (control for adaptivity).
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

VARIANTS = ["fuzzy", "fixed_1e-3", "fixed_mean_fuzzy",
            "fuzzy_inverted", "fuzzy_shuffled_vol"]


def _batch_index_list(n, batch_size):
    """Contiguous batches INCLUDING the final partial batch."""
    idx = np.arange(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def _variant_lr(variant, batch_vol, rng, vr_train=None):
    if variant == "fuzzy":
        return fz.fuzzy_lr(batch_vol)
    if variant == "fixed_1e-3":
        return cfg.FIXED_LR
    if variant == "fixed_mean_fuzzy":
        return fz.MEAN_LR
    if variant == "fuzzy_inverted":
        return fz.fuzzy_lr(batch_vol, invert=True)
    if variant == "fuzzy_shuffled_vol":
        # Same LR distribution (empirical train vol_rank -> fuzzy curve),
        # decoupled from this batch's own volatility.
        u = float(rng.choice(vr_train))
        return fz.fuzzy_lr(u)
    raise ValueError(variant)


def train_variant(X_train, y_train, vr_train, X_val, y_val,
                  variant, seed, epochs=cfg.EPOCHS,
                  batch_size=cfg.BATCH_SIZE, logdir="logs"):
    """Train one variant/seed. Returns (model, history, batch_log_df, info)."""
    cfg.set_seed(seed)
    os.makedirs(logdir, exist_ok=True)
    device = cfg.resolve_device()

    model = DirectionLSTM(n_features=X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.FIXED_LR)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    batches = _batch_index_list(len(Xt), batch_size)
    rng = np.random.default_rng(seed)

    # Regime-batching quantity: std of batch vol (fixed batching, pre-permutation)
    bvols = np.array([vr_train[b].mean() for b in batches])
    vol_std = float(bvols.std())

    rows = []
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(batches))  # permute batch ORDER only
        model.train()
        run_loss, n_seen = 0.0, 0
        for bi in order:
            idx = batches[bi]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            bvol = float(vr_train[idx].mean())
            lr = _variant_lr(variant, bvol, rng, vr_train)
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
    pd.DataFrame({"train_loss": history["train_loss"],
                  "val_loss": history["val_loss"],
                  "val_acc": history["val_acc"]}).to_csv(
        os.path.join(logdir, f"{variant}_s{seed}_hist.csv"), index=False)

    # corr(vol, lr) — NOTE: tautological w.r.t. training benefit; it only
    # checks the LR schedule was applied, not that it helped the LSTM.
    lr_std = float(log_df["lr"].std())
    # NOTE: tolerance (not == 0): pandas std of a constant float column can be
    # ~1e-19 from cancellation error, which would otherwise yield garbage ~0 corr.
    corr = float(log_df["vol"].corr(log_df["lr"])) if lr_std > 1e-12 else float("nan")
    info = {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "batch_vol_std": vol_std, "vol_lr_corr": corr}
    print(f"[{variant} s{seed}] best_epoch={best_epoch} val_loss={best_loss:.4f} "
          f"vol_std={vol_std:.3f} corr(vol,lr)={corr:.3f}")
    return model, history, log_df, info
