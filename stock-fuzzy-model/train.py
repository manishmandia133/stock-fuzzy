"""Training loop with regime-aware batching + five learning-rate variants.

Batching: train windows pre-sorted by end_date; each epoch permutes the ORDER
of contiguous batches (never shuffles samples across batches). The final
partial batch is INCLUDED (no silent drop_last). The same seed produces the
same model initialisation and batch order for every variant (paired design).

Methods (only these five are active):
  fixed_3e-4 / fixed_1e-3 / fixed_3e-3 - constant learning-rate baselines
  fuzzy_dyn - per-epoch Mamdani controller using relative val-loss change and
              the epoch mean vol_rank, with a 3-epoch linear warmup
  fuzzy_vol - Mamdani vol_rank -> LR curve applied per batch (ablation)

Early stopping uses the 3-epoch moving average of validation loss. Epochs
1..9 are never eligible to be restored (best_epoch >= MIN_EPOCHS). The best
eligible weights are restored. mean_lr_used (actual mean LR over all optimizer
steps) is recorded for every run.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

import config as cfg
import fuzzy as fz
from model import DirectionLSTM

VARIANTS = list(cfg.VARIANTS)
FIXED_LRS = dict(cfg.FIXED_LRS)
FIXED_VARIANTS = list(FIXED_LRS)
FUZZY_VARIANTS = list(cfg.FUZZY_VARIANTS)


def _batch_index_list(n, batch_size):
    """Contiguous batches INCLUDING the final partial batch."""
    idx = np.arange(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def _moving_average(values, window=cfg.EARLY_STOP_MA):
    """Causal moving average (partial window at the start)."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.mean(values[max(0, i - window + 1):i + 1])
    return out


def _variant_lr(variant, batch_vol):
    """Per-batch LR for the variants that are not epoch-driven."""
    if variant in FIXED_LRS:
        return FIXED_LRS[variant]
    if variant == "fuzzy_vol":
        return fz.fuzzy_vol_lr(batch_vol)
    if variant == "fuzzy_dyn":
        raise ValueError("fuzzy_dyn LR is set per epoch, not per batch")
    raise ValueError(f"unknown variant: {variant!r}")


def _validate_lrs(variant, lrs_used):
    """Raise if the applied LRs violate the method's contract."""
    lrs = np.asarray(lrs_used, dtype=float)
    if len(lrs) == 0:
        raise AssertionError(f"{variant}: no optimizer steps recorded")
    if variant in FIXED_LRS:
        expected = FIXED_LRS[variant]
        if not np.all(lrs == expected):
            raise AssertionError(
                f"{variant} changed LR: min={lrs.min():.12g} "
                f"max={lrs.max():.12g} expected={expected:.12g}")
        lr_std = 0.0  # mathematically exact; numpy cancellation gives ~1e-19
    elif variant in FUZZY_VARIANTS:
        if lrs.min() < cfg.LR_MIN - 1e-12 or lrs.max() > cfg.LR_MAX + 1e-12:
            raise AssertionError(
                f"{variant} LR out of bounds [{cfg.LR_MIN}, {cfg.LR_MAX}]: "
                f"min={lrs.min():.12g} max={lrs.max():.12g}")
        lr_std = float(lrs.std())
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return {"lr_min_used": float(lrs.min()),
            "lr_max_used": float(lrs.max()),
            "lr_std_used": lr_std}


def train_variant(X_train, y_train, vr_train, X_val, y_val,
                  variant, seed, epochs=None,
                  batch_size=cfg.BATCH_SIZE, logdir="logs"):
    """Train one variant/seed. Returns (model, history, batch_log_df, info)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}")
    epochs = cfg.EPOCHS if epochs is None else epochs
    if epochs < cfg.MIN_EPOCHS:
        raise ValueError(
            f"epochs={epochs} < MIN_EPOCHS={cfg.MIN_EPOCHS}; cannot restore "
            f"a best_epoch >= {cfg.MIN_EPOCHS}")
    cfg.set_seed(seed)
    os.makedirs(logdir, exist_ok=True)
    device = cfg.resolve_device()

    model = DirectionLSTM(n_features=X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.FIXED_LR,
                                 weight_decay=cfg.WEIGHT_DECAY)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    batches = _batch_index_list(len(Xt), batch_size)
    rng = np.random.default_rng(seed)

    # Regime-batching quantity: std of batch vol (fixed batching, pre-permutation)
    bvols = np.array([vr_train[b].mean() for b in batches])
    vol_std = float(bvols.std())

    # fuzzy_dyn uses one vol_rank input per epoch (mean over the training set).
    epoch_vol = float(np.mean(vr_train)) if len(vr_train) else 0.0
    controller = fz.FuzzyDynController(base_lr=cfg.FIXED_LR,
                                       warmup_epochs=cfg.WARMUP_EPOCHS)
    prev_vloss = None

    rows = []
    lrs_used = []  # actual LR applied at every optimizer step (fairness)
    history = {"train_loss": [], "val_loss": [], "val_loss_ma3": [],
               "val_acc": [], "val_auc": [], "lr_epoch": []}
    best_ma, best_state, best_epoch, stale = np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        # ---- learning rate for this epoch --------------------------------
        if variant == "fuzzy_dyn":
            epoch_lr = (controller.warmup_lr(epoch)
                        if epoch <= cfg.WARMUP_EPOCHS else controller.lr)
        else:
            epoch_lr = None

        order = rng.permutation(len(batches))  # permute batch ORDER only
        model.train()
        run_loss, n_seen = 0.0, 0
        epoch_lrs = []
        for bi in order:
            idx = batches[bi]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            if variant == "fuzzy_dyn":
                lr = epoch_lr
            else:
                # volatility of THIS actual training batch (train split only)
                bvol = float(vr_train[idx].mean())
                lr = _variant_lr(variant, bvol)
            lrs_used.append(lr)
            epoch_lrs.append(lr)
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
            log_row = {"epoch": epoch, "batch": int(bi), "lr": lr,
                       "loss": loss.item()}
            if variant != "fuzzy_dyn":
                log_row["vol"] = bvol
            else:
                log_row["vol"] = epoch_vol
            rows.append(log_row)

        # ---- validation ---------------------------------------------------
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            vloss = nn.BCELoss()(pv, yv).item()
            pv_np = pv.detach().cpu().numpy()
            vacc = (((pv > 0.5).float() == yv).sum().item() / max(len(yv), 1))
            vauc = float(roc_auc_score(y_val, pv_np)) if len(np.unique(y_val)) > 1 else float("nan")
        history["train_loss"].append(run_loss / max(n_seen, 1))
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        history["val_auc"].append(vauc)
        history["lr_epoch"].append(float(np.mean(epoch_lrs)))

        # Early-stopping signal: 3-epoch moving-average validation loss.
        ma = float(np.mean(history["val_loss"][-cfg.EARLY_STOP_MA:]))
        history["val_loss_ma3"].append(ma)

        # ---- fuzzy_dyn update (causal: applies to the NEXT epoch) ---------
        if variant == "fuzzy_dyn":
            if prev_vloss is not None and epoch >= cfg.WARMUP_EPOCHS:
                controller.update(vloss, epoch_vol)
            prev_vloss = float(vloss)

        # ---- early stopping (never restore epochs 1..MIN_EPOCHS-1) --------
        if epoch >= cfg.MIN_EPOCHS:
            if ma < best_ma - 1e-12:
                best_ma = ma
                best_state = copy.deepcopy(model.state_dict())
                best_epoch, stale = epoch, 0
            else:
                stale += 1
                if stale >= cfg.PATIENCE:
                    print(f"[{variant} s{seed}] early stop at epoch {epoch} "
                          f"(best MA3 {best_ma:.6f} at {best_epoch})")
                    break

    if best_state is None:
        raise AssertionError(
            f"{variant} s{seed}: no best weights at epoch >= {cfg.MIN_EPOCHS}")
    model.load_state_dict(best_state)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(logdir, f"{variant}_s{seed}.csv"), index=False)
    pd.DataFrame({"train_loss": history["train_loss"],
                  "val_loss": history["val_loss"],
                  "val_loss_ma3": history["val_loss_ma3"],
                  "val_acc": history["val_acc"],
                  "val_auc": history["val_auc"],
                  "lr_epoch": history["lr_epoch"]}).to_csv(
        os.path.join(logdir, f"{variant}_s{seed}_hist.csv"), index=False)

    # corr(vol, lr): meaningful only for fuzzy_vol; recorded for all variants.
    lr_std = float(log_df["lr"].std())
    vol_std_log = float(log_df["vol"].std())
    corr = (float(log_df["vol"].corr(log_df["lr"]))
            if lr_std > 1e-12 and vol_std_log > 1e-12 else float("nan"))
    mean_lr_used = float(np.mean(lrs_used)) if lrs_used else float("nan")
    lr_checks = _validate_lrs(variant, lrs_used)

    best_val_loss = float(history["val_loss"][best_epoch - 1])
    best_val_auc = float(history["val_auc"][best_epoch - 1])
    info = {"best_epoch": int(best_epoch),
            "best_val_loss": best_val_loss,
            "best_val_ma3": float(best_ma),
            "best_val_auc": best_val_auc,
            "epochs_run": len(history["val_loss"]),
            "early_stop_metric": f"val_loss_ma{cfg.EARLY_STOP_MA}",
            "batch_vol_std": vol_std, "vol_lr_corr": corr,
            "epoch_mean_vol": epoch_vol,
            "mean_lr_used": mean_lr_used, "n_steps": len(lrs_used),
            **lr_checks,
            # reproducibility / experimental settings
            "lr_policy": variant,
            "lr_init": float(FIXED_LRS.get(variant, cfg.FIXED_LR)),
            "lr_min": float(cfg.LR_MIN), "lr_max": float(cfg.LR_MAX),
            "warmup_epochs": int(cfg.WARMUP_EPOCHS),
            "window": int(cfg.WINDOW), "batch_size": int(batch_size),
            "max_epochs": int(epochs), "patience": int(cfg.PATIENCE),
            "min_epochs": int(cfg.MIN_EPOCHS),
            "grad_clip": float(cfg.GRAD_CLIP),
            "weight_decay": float(cfg.WEIGHT_DECAY),
            "train_frac": float(cfg.TRAIN_FRAC),
            "val_frac": float(cfg.VAL_FRAC),
            "universe_mode": cfg.UNIVERSE_MODE,
            "target_mode": cfg.TARGET_MODE,
            "n_features": int(X_train.shape[2]),
            "n_train": len(X_train), "n_val": len(X_val)}
    print(f"[{variant} s{seed}] best_epoch={best_epoch} "
          f"val_auc={best_val_auc:.4f} val_loss={best_val_loss:.4f} "
          f"MA3={best_ma:.6f} vol_std={vol_std:.3f} corr(vol,lr)={corr:.3f} "
          f"mean_lr_used={mean_lr_used:.3e} lr=[{lr_checks['lr_min_used']:.3e}, "
          f"{lr_checks['lr_max_used']:.3e}]")
    return model, history, log_df, info
