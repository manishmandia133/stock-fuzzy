"""Training loop with regime-aware batching + 2 fixed + 2 fuzzy methods.

Batching: train windows pre-sorted by end_date; each epoch permutes the ORDER
of contiguous batches (never shuffles samples across batches). The final
partial batch is INCLUDED (no silent drop_last).

Variants:
  2 fixed-LR grid baselines (see cfg.FIXED_LR_VARIANTS):
    fixed_2e-4, fixed_5e-4
  fuzzy          - two-input Mamdani LR: (batch vol_rank, epoch loss_trend)
  fuzzy_vol_only - ablation: previous one-input volatile-only controller

The loss trend is computed once per epoch from VALIDATION loss history and
held constant for every batch of the next epoch. Epochs 1-2 use trend 0.
mean_lr_used / lr_min_used / lr_max_used and the trend schedule are recorded.
"""

import copy
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

import config as cfg
import fuzzy as fz
from model import DirectionLSTM

FIXED_VARIANTS = list(cfg.FIXED_LR_VARIANTS)
FUZZY_VARIANTS = ["fuzzy", "fuzzy_vol_only"]
VARIANTS = FIXED_VARIANTS + FUZZY_VARIANTS

INITIAL_LR = cfg.FIXED_LR
EARLY_STOP_TOL = 1e-9
CORR_EPS = 1e-12
TREND_EPS = 1e-8


def _batch_index_list(n: int, batch_size: int) -> list[np.ndarray]:
    """Contiguous batches INCLUDING the final partial batch."""
    idx = np.arange(n)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _variant_lr(variant: str, batch_vol: float, trend: float) -> float:
    """Resolve the learning rate for one optimizer step."""
    if variant in cfg.FIXED_LR_VARIANTS:
        return cfg.FIXED_LR_VARIANTS[variant]
    if variant == "fuzzy":
        return fz.fuzzy_lr(batch_vol, trend)
    if variant == "fuzzy_vol_only":
        return fz.fuzzy_lr_vol_only(batch_vol)
    raise ValueError(f"unknown or removed variant: {variant!r}")


def _validate_lrs(
    variant: str,
    lrs_used: Sequence[float],
) -> dict[str, float]:
    """Raise if the applied LRs violate the method's contract."""
    lrs = np.asarray(lrs_used, dtype=float)
    if len(lrs) == 0:
        raise AssertionError(f"{variant}: no optimizer steps recorded")
    if variant in cfg.FIXED_LR_VARIANTS:
        expected = cfg.FIXED_LR_VARIANTS[variant]
        if not np.all(lrs == expected):
            raise AssertionError(
                f"{variant} changed LR: min={lrs.min():.12g} "
                f"max={lrs.max():.12g} expected={expected:.12g}"
            )
        lr_std = 0.0  # mathematically exact; numpy cancellation gives ~1e-19
    elif variant in FUZZY_VARIANTS:
        if lrs.min() < cfg.LR_MIN - 1e-12 or lrs.max() > cfg.LR_MAX + 1e-12:
            raise AssertionError(
                f"{variant} LR out of bounds [{cfg.LR_MIN}, {cfg.LR_MAX}]: "
                f"min={lrs.min():.12g} max={lrs.max():.12g}"
            )
        lr_std = float(lrs.std())
    else:
        raise ValueError(f"unknown or removed variant: {variant!r}")
    return {
        "lr_min_used": float(lrs.min()),
        "lr_max_used": float(lrs.max()),
        "lr_std_used": lr_std,
    }


def _next_trend(history: dict[str, list[float]], epoch: int) -> float:
    """Validation-loss trend for the NEXT epoch (epochs 1-2 use trend 0)."""
    if epoch == 1:
        return 0.0
    vals = np.asarray(history["val_loss"], dtype=float)
    if len(vals) < 2:
        return 0.0
    base = float(np.mean(vals[-3:-1]))
    raw = (vals[-1] - base) / (abs(base) + TREND_EPS)
    return float(np.clip(raw / cfg.TREND_SCALE, cfg.TREND_MIN, cfg.TREND_MAX))


def _prepare_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to torch tensors with validation already on-device."""
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
    return Xt, yt, Xv, yv


def _run_epoch(
    model: DirectionLSTM,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    batches: list[np.ndarray],
    order: np.ndarray,
    vr_train: np.ndarray,
    variant: str,
    epoch: int,
    epoch_trend: float,
    device: torch.device | str,
) -> tuple[float, int, list[dict[str, float | int]], list[float]]:
    """Run one epoch in the supplied batch order and return its accounting."""
    model.train()
    run_loss, n_seen = 0.0, 0
    rows: list[dict[str, float | int]] = []
    lrs: list[float] = []
    for bi in order:
        idx = batches[bi]
        xb, yb = Xt[idx].to(device), yt[idx].to(device)
        # volatility of THIS actual training batch (train split only)
        bvol = float(vr_train[idx].mean())
        lr = _variant_lr(variant, bvol, epoch_trend)
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        loss_value = loss.item()
        run_loss += loss_value * len(idx)
        n_seen += len(idx)
        rows.append(
            {
                "epoch": epoch,
                "batch": int(bi),
                "vol": bvol,
                "trend": epoch_trend,
                "lr": lr,
                "loss": loss_value,
            }
        )
    return run_loss, n_seen, rows, lrs


def _validate_epoch(
    model: DirectionLSTM,
    Xv: torch.Tensor,
    yv: torch.Tensor,
) -> tuple[float, float]:
    """Compute validation BCE loss and accuracy."""
    model.eval()
    with torch.no_grad():
        pv = model(Xv)
        vloss = nn.BCELoss()(pv, yv).item()
        vacc = ((pv > 0.5).float() == yv).sum().item() / max(len(yv), 1)
    return float(vloss), float(vacc)


def _save_logs(
    logdir: str,
    variant: str,
    seed: int,
    rows: list[dict[str, float | int]],
    history: dict[str, list[float]],
) -> pd.DataFrame:
    """Write per-step and per-epoch logs; return the per-step frame."""
    os.makedirs(logdir, exist_ok=True)
    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(logdir, f"{variant}_s{seed}.csv"), index=False)
    pd.DataFrame(
        {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
            "trend_used": history["trend_used"],
        }
    ).to_csv(os.path.join(logdir, f"{variant}_s{seed}_hist.csv"), index=False)
    return log_df


def _build_info(
    variant: str,
    best_epoch: int,
    best_loss: float,
    vol_std: float,
    corr: float,
    mean_lr_used: float,
    n_steps: int,
    trends: np.ndarray,
    lr_checks: dict[str, float],
    batch_size: int,
    epochs: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> dict[str, float | int | str]:
    """Assemble the run metadata written to results.csv."""
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "batch_vol_std": vol_std,
        "vol_lr_corr": corr,
        "mean_lr_used": mean_lr_used,
        "n_steps": n_steps,
        "trend_min_used": float(trends.min()) if len(trends) else 0.0,
        "trend_max_used": float(trends.max()) if len(trends) else 0.0,
        "trend_std_used": float(trends.std()) if len(trends) else 0.0,
        "mean_trend_used": float(trends.mean()) if len(trends) else 0.0,
        **lr_checks,
        # reproducibility / experimental settings
        "lr_policy": variant,
        "lr_init": float(cfg.FIXED_LR_VARIANTS.get(variant, cfg.LR_MED_C)),
        "lr_min": float(cfg.LR_MIN),
        "lr_max": float(cfg.LR_MAX),
        "window": int(cfg.WINDOW),
        "hidden": int(cfg.HIDDEN),
        "num_layers": int(cfg.NUM_LAYERS),
        "dropout": float(cfg.DROPOUT),
        "batch_size": int(batch_size),
        "max_epochs": int(epochs),
        "patience": int(cfg.PATIENCE),
        "grad_clip": float(cfg.GRAD_CLIP),
        "train_frac": float(cfg.TRAIN_FRAC),
        "val_frac": float(cfg.VAL_FRAC),
        "n_features": int(X_train.shape[2]),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def _print_run_info(
    variant: str,
    seed: int,
    info: dict[str, float | int | str],
    corr: float,
) -> None:
    """Print the one-line run summary."""
    print(
        f"[{variant} s{seed}] best_epoch={info['best_epoch']} "
        f"val_loss={info['best_val_loss']:.4f} "
        f"vol_std={info['batch_vol_std']:.3f} corr(vol,lr)={corr:.3f} "
        f"mean_lr_used={info['mean_lr_used']:.3e} "
        f"lr=[{info['lr_min_used']:.3e}, {info['lr_max_used']:.3e}] "
        f"trend=[{info['trend_min_used']:.3f}, {info['trend_max_used']:.3f}]"
    )


def _train_epochs(
    model: DirectionLSTM,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    Xv: torch.Tensor,
    yv: torch.Tensor,
    batches: list[np.ndarray],
    vr_train: np.ndarray,
    variant: str,
    seed: int,
    epochs: int,
    device: torch.device | str,
    rng: np.random.Generator,
) -> tuple[
    dict[str, torch.Tensor] | None,
    int,
    float,
    list[dict[str, float | int]],
    list[float],
    list[float],
    dict[str, list[float]],
]:
    """Run the epoch loop and return best checkpoint plus full accounting."""
    rows: list[dict[str, float | int]] = []
    lrs_used: list[float] = []  # actual LR applied at every optimizer step
    trends_used: list[float] = []
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "trend_used": [],
    }
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    epoch_trend = 0.0  # epochs 1-2 use 0; later use previous epoch's val trend

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(batches))  # permute batch ORDER only
        trends_used.append(epoch_trend)
        run_loss, n_seen, epoch_rows, epoch_lrs = _run_epoch(
            model,
            optimizer,
            criterion,
            Xt,
            yt,
            batches,
            order,
            vr_train,
            variant,
            epoch,
            epoch_trend,
            device,
        )
        rows.extend(epoch_rows)
        lrs_used.extend(epoch_lrs)

        # validation (also drives next epoch's trend)
        vloss, vacc = _validate_epoch(model, Xv, yv)
        history["train_loss"].append(run_loss / max(n_seen, 1))
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        history["trend_used"].append(epoch_trend)
        epoch_trend = _next_trend(history, epoch)

        if vloss < best_loss - EARLY_STOP_TOL:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= cfg.PATIENCE:
                print(
                    f"[{variant} s{seed}] early stop at epoch {epoch} "
                    f"(best {best_epoch})"
                )
                break
    return best_state, best_epoch, best_loss, rows, lrs_used, trends_used, history


def train_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vr_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    variant: str,
    seed: int,
    epochs: int = cfg.EPOCHS,
    batch_size: int = cfg.BATCH_SIZE,
    logdir: str = "logs",
) -> tuple[
    DirectionLSTM,
    dict[str, list[float]],
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """Train one variant/seed. Returns (model, history, batch_log_df, info)."""
    cfg.set_seed(seed)
    os.makedirs(logdir, exist_ok=True)
    device = cfg.resolve_device()

    model = DirectionLSTM(n_features=X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    Xt, yt, Xv, yv = _prepare_tensors(X_train, y_train, X_val, y_val, device)

    batches = _batch_index_list(len(Xt), batch_size)
    rng = np.random.default_rng(seed)

    # Regime-batching quantity: std of batch vol (fixed batching, pre-permutation).
    bvols = np.array([vr_train[b].mean() for b in batches])
    vol_std = float(bvols.std())

    (
        best_state,
        best_epoch,
        best_loss,
        rows,
        lrs_used,
        trends_used,
        history,
    ) = _train_epochs(
        model,
        optimizer,
        criterion,
        Xt,
        yt,
        Xv,
        yv,
        batches,
        vr_train,
        variant,
        seed,
        epochs,
        device,
        rng,
    )

    model.load_state_dict(best_state)
    log_df = _save_logs(logdir, variant, seed, rows, history)

    # corr(vol, lr) — NOTE: tautological w.r.t. training benefit; it only
    # checks the LR schedule was applied, not that it helped the LSTM.
    lr_std = float(log_df["lr"].std())
    # NOTE: tolerance (not == 0): pandas std of a constant float column can be
    # ~1e-19 from cancellation error, which would otherwise yield garbage ~0 corr.
    corr = (
        float(log_df["vol"].corr(log_df["lr"])) if lr_std > CORR_EPS else float("nan")
    )
    mean_lr_used = float(np.mean(lrs_used)) if lrs_used else float("nan")
    lr_checks = _validate_lrs(variant, lrs_used)
    trends = np.asarray(trends_used, dtype=float)
    info = _build_info(
        variant,
        best_epoch,
        best_loss,
        vol_std,
        corr,
        mean_lr_used,
        len(lrs_used),
        trends,
        lr_checks,
        batch_size,
        epochs,
        X_train,
        X_val,
    )
    _print_run_info(variant, seed, info, corr)
    return model, history, log_df, info
