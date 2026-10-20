"""Evaluation: AUC, acc, MCC, majority baseline + stability + bootstrap CI."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

import config as cfg

ROLLING_WINDOW = 3
HIGH_VOL_CUTOFF = 0.7
LEAKAGE_AUC = 0.60
LOGREG_MAX_ITER = 1000
SUMMARY_METRICS = [
    "test_auc",
    "test_acc",
    "test_mcc",
    "best_val_loss",
    "highvol_loss_var",
    "train_val_gap",
    "mean_lr_used",
    "vol_lr_corr",
]


@torch.no_grad()
def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for X without tracking gradients."""
    model.eval()
    device = cfg.resolve_device()
    return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()


def pd_roll(val: Sequence[float]) -> np.ndarray:
    """Rolling mean with the configured window and min_periods=1."""
    return pd.Series(val).rolling(ROLLING_WINDOW, min_periods=1).mean().values


def stability_metrics(
    history: dict[str, list[float]],
    log_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Compute loss-spike, best-epoch, generalization-gap, and high-vol variance."""
    val = np.array(history["val_loss"])
    train = np.array(history["train_loss"])
    roll = pd_roll(val)
    spikes = int(((val[1:] - roll[:-1]) > val.std()).sum()) if len(val) > 2 else 0
    best = int(np.argmin(val)) + 1
    gap = float(val[best - 1] - train[best - 1])  # val_loss - train_loss
    if "vol" in log_df:
        hv = float(log_df.loc[log_df["vol"] > HIGH_VOL_CUTOFF, "loss"].var())
    else:
        hv = float("nan")
    return {
        "spikes": spikes,
        "epochs_to_best": best,
        "train_val_gap": gap,
        "highvol_loss_var": hv,
    }


def bootstrap_auc_ci(
    y_true: Sequence[float],
    proba: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for ROC AUC (percentile method)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], proba[idx]))
        except Exception:  # noqa: BLE001, S112
            continue
    aucs = np.array(aucs)
    if len(aucs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_run(
    model,
    history: dict[str, list[float]],
    log_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_last: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float | bool | int]:
    """Evaluate one trained run and return report-ready metrics."""
    proba = predict_proba(model, X_test)
    pred = (proba > 0.5).astype(int)
    auc = (
        float(roc_auc_score(y_test, proba))
        if len(np.unique(y_test)) > 1
        else float("nan")
    )
    lo, hi = bootstrap_auc_ci(y_test, proba, seed=seed)
    pos_rate = float(pred.mean())
    pstd = float(proba.std())
    # Constant-predictor flag: model outputs a single class at 0.5 threshold.
    const_flag = bool(pos_rate == 0.0 or pos_rate == 1.0)
    out: dict[str, float | bool | int] = {
        "test_auc": auc,
        "test_auc_lo": lo,
        "test_auc_hi": hi,
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_mcc": float(matthews_corrcoef(y_test, pred)),
        "majority_acc": float(max(y_test.mean(), 1 - y_test.mean())),
        "pred_pos_rate": pos_rate,
        "pred_std": pstd,
        "const_pred_flag": const_flag,
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": float(np.min(history["val_loss"])),
    }
    out.update(stability_metrics(history, log_df))

    # Gate: AUC > 0.60 suspect leakage (flag, do not silently report).
    out["leakage_flag"] = bool(out["test_auc"] > LEAKAGE_AUC)

    # Gate: logistic regression on flattened last-day features ~ 0.50 AUC.
    if X_train_last is not None and y_train is not None:
        logreg = LogisticRegression(max_iter=LOGREG_MAX_ITER).fit(X_train_last, y_train)
        out["logreg_auc"] = float(
            roc_auc_score(y_test, logreg.predict_proba(X_test[:, -1, :])[:, 1])
        )
    return out


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float, int]:
    """Paired t-test and Wilcoxon test over finite observations."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[mask], b_arr[mask]
    if len(a_arr) < 2:
        return float("nan"), float("nan"), len(a_arr)
    try:
        _, p_t = ttest_rel(a_arr, b_arr)
    except Exception:  # noqa: BLE001
        p_t = float("nan")
    try:
        _, p_w = wilcoxon(a_arr, b_arr)
    except Exception:  # noqa: BLE001
        p_w = float("nan")
    return float(p_t), float(p_w), len(a_arr)


def select_best_fixed(res: pd.DataFrame) -> tuple[str, pd.Series]:
    """Select BEST_FIXED by lowest mean validation loss (no test metrics)."""
    fixed = res[res["variant"].isin(cfg.FIXED_LR_VARIANTS)]
    means = fixed.groupby("variant")["best_val_loss"].mean().sort_values()
    return str(means.index[0]), means


def summary_table(res: pd.DataFrame, variants: Sequence[str]) -> pd.DataFrame:
    """Compact mean ± std table for the report."""
    parts = []
    for variant in variants:
        sub = res[res["variant"] == variant]
        row: dict[str, float | int | str] = {"variant": variant}
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1))
        row["n"] = len(sub)
        parts.append(row)
    return pd.DataFrame(parts)
