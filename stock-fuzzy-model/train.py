"""Training loop for ^GSPC regression.

Supported variants (strictly fuzzy models):
  Fuzzy_MoE          – Takagi-Sugeno / Ruspini Fuzzy-gated MoE (3 regime experts)
  Fuzzy_Input        – Neural Fuzzy with Ruspini membership degrees as inputs
  Fuzzy_Adaptive_MoE – Fuzzy MoE + fuzzy adaptive gradient clipping
  Fuzzy_ANFIS        – Hybrid ANFIS architecture (Fuzzy inputs + Fuzzy MoE)

Returns (model, history, val_preds_np, test_preds_np, info).
val_preds / test_preds are predictions as log-return arrays (numpy).
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

import config as cfg
import fuzzy as fz
from model import build_model

VARIANTS = cfg.VARIANTS


def _make_criterion():
    if cfg.LOSS == "huber":
        return nn.HuberLoss(delta=cfg.HUBER_DELTA)
    return nn.MSELoss()


def _get_lr(variant: str, vr: float, tr: float) -> float:
    # All variants use the Mamdani fuzzy LR controller
    return fz.fuzzy_lr(vr)


def _get_clip(variant: str, vr: float) -> float:
    if "Adaptive" in variant:
        # High vol → tighter clip (reduce noise in gradient)
        return float(np.clip(cfg.GRAD_CLIP * (1.0 - 0.5 * vr), 0.1, cfg.GRAD_CLIP))
    return cfg.GRAD_CLIP


def _batch_indices(n: int, batch_size: int):
    return [np.arange(n)[i:i + batch_size] for i in range(0, n, batch_size)]


def _moving_avg(vals, w=cfg.EARLY_STOP_MA):
    out, v = [], np.asarray(vals, float)
    for i in range(len(v)):
        out.append(float(np.mean(v[max(0, i - w + 1):i + 1])))
    return out


@torch.no_grad()
def _predict(model, X_np, vr_np, tr_np, device):
    model.eval()
    Xt  = torch.tensor(X_np,  dtype=torch.float32).to(device)
    vrt = torch.tensor(vr_np, dtype=torch.float32).to(device)
    trt = torch.tensor(tr_np, dtype=torch.float32).to(device)
    return model(Xt, vr=vrt, tr=trt).cpu().numpy()


def train_model(data: dict,
                variant: str = "M2",
                seed: int = 0,
                epochs: int = cfg.EPOCHS,
                batch_size: int = cfg.BATCH_SIZE,
                logdir: str = "logs"):
    """Train one variant for one seed.

    Returns (model, history, val_preds_np, test_preds_np, info).
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of {VARIANTS}")

    cfg.set_seed(seed)
    os.makedirs(logdir, exist_ok=True)
    device = cfg.resolve_device()

    n_features = data["X_train"].shape[2]
    model      = build_model(variant, n_features,
                             hidden_size=cfg.HIDDEN_SIZE,
                             num_layers=cfg.NUM_LAYERS,
                             dropout=cfg.DROPOUT).to(device)
    criterion  = _make_criterion()
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=cfg.INITIAL_LR,
                                  weight_decay=cfg.WEIGHT_DECAY)

    # Tensors
    Xt_all  = torch.tensor(data["X_train"],    dtype=torch.float32)
    yt_all  = torch.tensor(data["y_train"],    dtype=torch.float32)
    Xv      = torch.tensor(data["X_val"],      dtype=torch.float32).to(device)
    yv      = torch.tensor(data["y_val"],      dtype=torch.float32).to(device)
    vr_v    = torch.tensor(data["vr_val"],     dtype=torch.float32).to(device)
    tr_v    = torch.tensor(data["tr_val"],     dtype=torch.float32).to(device)

    vr_train = data["vr_train"]
    tr_train = data["tr_train"]

    batches = _batch_indices(len(Xt_all), batch_size)
    rng     = np.random.default_rng(seed)

    history = {"train_loss": [], "val_loss": [], "val_loss_ma": [],
               "lr_epoch": [], "grad_clip_epoch": []}
    batch_log = []

    best_ma, best_state, best_epoch, stale = np.inf, None, 0, 0
    lrs_used = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(batches))
        model.train()
        run_loss, n_seen = 0.0, 0
        epoch_lrs, epoch_clips = [], []

        for bi in order:
            idx  = batches[bi]
            xb   = Xt_all[idx].to(device)
            yb   = yt_all[idx].to(device)
            bvr  = float(vr_train[idx].mean())
            btr  = float(tr_train[idx].mean())

            lr   = _get_lr(variant, bvr, btr)
            clip = _get_clip(variant, bvr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            vrt_b = torch.tensor(vr_train[idx], dtype=torch.float32).to(device)
            trt_b = torch.tensor(tr_train[idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            pred = model(xb, vr=vrt_b, tr=trt_b)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            run_loss += loss.item() * len(idx)
            n_seen   += len(idx)
            lrs_used.append(lr)
            epoch_lrs.append(lr)
            epoch_clips.append(clip)
            batch_log.append({"epoch": epoch, "batch": int(bi),
                              "lr": lr, "clip": clip,
                              "loss": loss.item(), "vol": bvr})

        # Validation
        model.eval()
        with torch.no_grad():
            pv   = model(Xv, vr=vr_v, tr=tr_v)
            vloss = criterion(pv, yv).item()

        train_loss = run_loss / max(n_seen, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss)
        history["lr_epoch"].append(float(np.mean(epoch_lrs)))
        history["grad_clip_epoch"].append(float(np.mean(epoch_clips)))
        ma = float(np.mean(history["val_loss"][-cfg.EARLY_STOP_MA:]))
        history["val_loss_ma"].append(ma)

        if epoch >= cfg.MIN_EPOCHS:
            if ma < best_ma - 1e-12:
                best_ma    = ma
                best_state = copy.deepcopy(model.state_dict())
                best_epoch, stale = epoch, 0
            else:
                stale += 1
                if stale >= cfg.PATIENCE:
                    print(f"  [{variant} s{seed}] early stop ep{epoch} "
                          f"(best MA{cfg.EARLY_STOP_MA}={best_ma:.6f} "
                          f"at ep{best_epoch})")
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history["val_loss"])
        print(f"  [{variant} s{seed}] WARNING: no best state before MIN_EPOCHS; "
              "using final weights.")
    model.load_state_dict(best_state)

    # Save batch log
    tag = f"{variant}_s{seed}"
    pd.DataFrame(batch_log).to_csv(
        os.path.join(logdir, f"{tag}_batch.csv"), index=False)
    pd.DataFrame({k: history[k] for k in history}).to_csv(
        os.path.join(logdir, f"{tag}_hist.csv"), index=False)

    # Predictions on val and test
    val_preds  = _predict(model, data["X_val"],  data["vr_val"],  data["tr_val"],  device)
    test_preds = _predict(model, data["X_test"], data["vr_test"], data["tr_test"], device)

    lrs_arr = np.asarray(lrs_used, float)
    info = {
        "variant":         variant,
        "seed":            seed,
        "best_epoch":      best_epoch,
        "best_val_loss":   float(history["val_loss"][best_epoch - 1]),
        "best_val_ma":     float(best_ma),
        "epochs_run":      len(history["val_loss"]),
        "lr_mean":         float(lrs_arr.mean()) if len(lrs_arr) else float("nan"),
        "lr_std":          float(lrs_arr.std())  if len(lrs_arr) else float("nan"),
        "lr_min":          float(lrs_arr.min())  if len(lrs_arr) else float("nan"),
        "lr_max":          float(lrs_arr.max())  if len(lrs_arr) else float("nan"),
        "loss_fn":         cfg.LOSS,
        "window":          cfg.WINDOW,
        "batch_size":      batch_size,
        "hidden_size":     cfg.HIDDEN_SIZE,
        "num_layers":      cfg.NUM_LAYERS,
        "dropout":         cfg.DROPOUT,
    }
    best_tl = float(history["train_loss"][best_epoch - 1])
    best_vl = float(history["val_loss"][best_epoch - 1])
    print(f"  [{variant} s{seed}] ep={best_epoch}  "
          f"train_loss={best_tl:.5f}  val_loss={best_vl:.5f}  "
          f"lr_mean={info['lr_mean']:.3e}")

    return model, history, val_preds, test_preds, info
