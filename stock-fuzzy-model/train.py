"""Training loop with regime-aware batching + 2 learning-rate methods.

Batching: train windows pre-sorted by end_date; each epoch permutes the ORDER
of contiguous batches (never shuffles samples across batches). The final
partial batch is INCLUDED (no silent drop_last).

Methods (only these two are active):
  fixed_1e-3 - constant learning rate of 1e-3 throughout training (baseline)
  fuzzy      - Mamdani fuzzy LR driven by the ACTUAL training batch's mean
               vol_rank (proposed method)
mean_lr_used (actual mean LR over all optimizer steps) is recorded per run.
The chosen LR is validated on every optimizer step:
  - fixed_1e-3 must equal cfg.FIXED_LR exactly (never changes);
  - fuzzy must stay within [cfg.LR_MIN, cfg.LR_MAX].
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

VARIANTS = ["fixed_1e-3", "fuzzy"]


def _batch_index_list(n, batch_size):
    """Contiguous batches INCLUDING the final partial batch."""
    idx = np.arange(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def _variant_lr(variant, batch_vol):
    if variant == "fixed_1e-3":
        return cfg.FIXED_LR
    if variant == "fuzzy":
        return fz.fuzzy_lr(batch_vol)
    raise ValueError(f"unknown or removed variant: {variant!r}")


def _validate_lrs(variant, lrs_used):
    """Raise if the applied LRs violate the method's contract."""
    lrs = np.asarray(lrs_used, dtype=float)
    if len(lrs) == 0:
        raise AssertionError(f"{variant}: no optimizer steps recorded")
    if variant == "fixed_1e-3":
        if not np.all(lrs == cfg.FIXED_LR):
            raise AssertionError(
                f"fixed_1e-3 changed LR: min={lrs.min():.12g} "
                f"max={lrs.max():.12g} expected={cfg.FIXED_LR:.12g}")
        lr_std = 0.0  # mathematically exact; numpy cancellation gives ~1e-19
    elif variant == "fuzzy":
        if lrs.min() < cfg.LR_MIN - 1e-12 or lrs.max() > cfg.LR_MAX + 1e-12:
            raise AssertionError(
                f"fuzzy LR out of bounds [{cfg.LR_MIN}, {cfg.LR_MAX}]: "
                f"min={lrs.min():.12g} max={lrs.max():.12g}")
        lr_std = float(lrs.std())
    else:
        raise ValueError(f"unknown or removed variant: {variant!r}")
    return {"lr_min_used": float(lrs.min()),
            "lr_max_used": float(lrs.max()),
            "lr_std_used": lr_std}


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
    lrs_used = []  # actual LR applied at every optimizer step (RQ1 fairness)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(batches))  # permute batch ORDER only
        model.train()
        run_loss, n_seen = 0.0, 0
        for bi in order:
            idx = batches[bi]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            # volatility of THIS actual training batch (train split only)
            bvol = float(vr_train[idx].mean())
            lr = _variant_lr(variant, bvol)
            lrs_used.append(lr)
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
    mean_lr_used = float(np.mean(lrs_used)) if lrs_used else float("nan")
    lr_checks = _validate_lrs(variant, lrs_used)
    info = {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "batch_vol_std": vol_std, "vol_lr_corr": corr,
            "mean_lr_used": mean_lr_used, "n_steps": len(lrs_used),
            **lr_checks,
            # reproducibility / experimental settings
            "lr_policy": variant, "lr_init": float(cfg.FIXED_LR),
            "lr_min": float(cfg.LR_MIN), "lr_max": float(cfg.LR_MAX),
            "window": int(cfg.WINDOW), "batch_size": int(batch_size),
            "max_epochs": int(epochs), "patience": int(cfg.PATIENCE),
            "grad_clip": float(cfg.GRAD_CLIP),
            "train_frac": float(cfg.TRAIN_FRAC),
            "val_frac": float(cfg.VAL_FRAC),
            "n_features": int(X_train.shape[2]),
            "n_train": int(len(X_train)), "n_val": int(len(X_val))}
    print(f"[{variant} s{seed}] best_epoch={best_epoch} val_loss={best_loss:.4f} "
          f"vol_std={vol_std:.3f} corr(vol,lr)={corr:.3f} "
          f"mean_lr_used={mean_lr_used:.3e} lr=[{lr_checks['lr_min_used']:.3e}, "
          f"{lr_checks['lr_max_used']:.3e}]")
    return model, history, log_df, info
