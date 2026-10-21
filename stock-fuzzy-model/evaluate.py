"""Evaluation: primary test AUC, secondary metrics, paired stats.

Primary metric (fixed in advance): test ROC AUC.
Secondary metrics: test log-loss, epochs-to-target validation loss, area under
the validation-loss curve, test AUC on high-vol days (top tercile of vol_rank),
and across-seed std of the primary metric.
"""
import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

import config as cfg

PRIMARY_METRIC = "test_auc"
SECONDARY_METRICS = ["test_logloss", "epochs_to_target_val_loss",
                     "aulc_val_loss", "test_auc_highvol"]
COMPARE_METRICS = [PRIMARY_METRIC] + SECONDARY_METRICS


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        device = cfg.resolve_device()
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()


def select_threshold(y_val, proba_val):
    """Choose the VAL threshold (0.40..0.60, step 0.005) by balanced accuracy."""
    grid = np.arange(cfg.THRESHOLD_LOW, cfg.THRESHOLD_HIGH + 0.5 * cfg.THRESHOLD_STEP,
                     cfg.THRESHOLD_STEP)
    scores = np.array([
        balanced_accuracy_score(y_val, (proba_val > t).astype(int))
        for t in grid
    ])
    best = float(np.max(scores))
    candidates = grid[np.isclose(scores, best)]
    # Tie-break toward the neutral 0.5 threshold.
    thr = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return thr, float(np.max(scores))


def moving_average(values, window=cfg.EARLY_STOP_MA):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.mean(values[max(0, i - window + 1):i + 1])
    return out


def best_epoch_from_history(history):
    """First-eligible epoch with the lowest 3-epoch moving-average val loss."""
    ma = np.asarray(history.get("val_loss_ma3", moving_average(history["val_loss"])),
                    dtype=float)
    eligible = [i for i in range(len(ma)) if i + 1 >= cfg.MIN_EPOCHS]
    if not eligible:
        raise ValueError(f"no epoch >= MIN_EPOCHS={cfg.MIN_EPOCHS}")
    return int(min(eligible, key=lambda i: ma[i])) + 1


def epochs_to_target_val_loss(history, target):
    """First epoch whose 3-epoch MA val loss is <= target (censored at run end)."""
    if target is None or not np.isfinite(target):
        return float("nan")
    ma = np.asarray(history.get("val_loss_ma3", moving_average(history["val_loss"])),
                    dtype=float)
    hits = np.where(ma <= target)[0]
    return float(hits[0] + 1) if len(hits) else float(len(ma))


def area_under_val_loss_curve(history):
    """Trapezoidal area under the per-epoch validation-loss curve."""
    val = np.asarray(history["val_loss"], dtype=float)
    if len(val) < 2:
        return float("nan")
    # Trapezoidal rule; implemented directly for NumPy 1.x/2.x compatibility.
    return float(np.sum(0.5 * (val[1:] + val[:-1])))


def stability_metrics(history, log_df, best_epoch=None):
    val = np.array(history["val_loss"])
    train = np.array(history["train_loss"])
    roll = moving_average(val, 3)
    spikes = int(((val[1:] - roll[:-1]) > val.std()).sum()) if len(val) > 2 else 0
    if best_epoch is None:
        best = best_epoch_from_history(history)
    else:
        best = int(best_epoch)
    gap = float(val[best - 1] - train[best - 1])  # val_loss - train_loss
    hv = float(log_df.loc[log_df["vol"] > 0.7, "loss"].var()) if "vol" in log_df else float("nan")
    return {"spikes": spikes, "epochs_to_best": best,
            "train_val_gap": gap, "highvol_loss_var": hv}


def bootstrap_auc_ci(y_true, proba, n_boot=200, seed=0):
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


def _classification_metrics(y_true, pred):
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "pos_rate": float(np.mean(pred)),
    }


def evaluate_run(model, history, log_df, X_val, y_val, X_test, y_test,
                 X_train_last=None, y_train=None, seed=0,
                 vr_test=None, best_epoch=None, val_loss_target=None):
    proba_val = predict_proba(model, X_val)
    proba = predict_proba(model, X_test)

    thr, thr_bal_acc = select_threshold(y_val, proba_val)
    pred = (proba > thr).astype(int)
    m_thr = _classification_metrics(y_test, pred)
    m_05 = _classification_metrics(y_test, (proba > 0.5).astype(int))

    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    lo, hi = bootstrap_auc_ci(y_test, proba, seed=seed)
    test_logloss = float(log_loss(y_test, proba, labels=[0, 1]))

    highvol_auc = float("nan")
    if vr_test is not None and len(vr_test) == len(y_test):
        vol_cut = float(np.quantile(vr_test, 2.0 / 3.0))
        mask = np.asarray(vr_test) >= vol_cut
        if mask.sum() > 1 and len(np.unique(np.asarray(y_test)[mask])) > 1:
            highvol_auc = float(roc_auc_score(np.asarray(y_test)[mask],
                                              np.asarray(proba)[mask]))

    if best_epoch is None:
        best_epoch = best_epoch_from_history(history)
    best_epoch = int(best_epoch)
    val_auc = np.asarray(history["val_auc"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)

    out = {"test_auc": auc,
           "test_auc_lo": lo,
           "test_auc_hi": hi,
           "test_logloss": test_logloss,
           "test_auc_highvol": highvol_auc,
           "aulc_val_loss": area_under_val_loss_curve(history),
           "epochs_to_target_val_loss": epochs_to_target_val_loss(
               history, val_loss_target),
           "test_acc": m_thr["acc"],
           "test_mcc": m_thr["mcc"],
           "test_bal_acc": m_thr["bal_acc"],
           "majority_acc": float(max(y_test.mean(), 1 - y_test.mean())),
           "threshold_val": thr,
           "threshold_val_bal_acc": thr_bal_acc,
           "pred_pos_rate": m_thr["pos_rate"],
           "pred_std": float(proba.std()),
           "const_pred_flag": bool(m_thr["pos_rate"] == 0.0 or m_thr["pos_rate"] == 1.0),
           "test_acc_05": m_05["acc"],
           "test_mcc_05": m_05["mcc"],
           "pred_pos_rate_05": m_05["pos_rate"],
           "const_pred_flag_05": bool(m_05["pos_rate"] == 0.0 or m_05["pos_rate"] == 1.0),
           "best_epoch": best_epoch,
           "best_val_loss": float(val_loss[best_epoch - 1]),
           "best_val_auc": float(val_auc[best_epoch - 1]),
           "min_val_loss": float(np.min(val_loss))}
    out.update(stability_metrics(history, log_df, best_epoch=best_epoch))

    # Gate: AUC > 0.60 suspect leakage (flag, do not silently report)
    out["leakage_flag"] = bool(out["test_auc"] > 0.60)

    # Gate: logistic regression on flattened last-day features ~ 0.50 AUC
    if X_train_last is not None and y_train is not None:
        lr = LogisticRegression(max_iter=1000).fit(X_train_last, y_train)
        out["logreg_auc"] = float(roc_auc_score(
            y_test, lr.predict_proba(X_test[:, -1, :])[:, 1]))
    return out


def paired_bootstrap_ci(a, b, n_boot=10000, seed=0):
    """95% percentile bootstrap CI for mean(a - b), paired by row."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(n_boot)])
    return (float(diff.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def paired_tests(a, b):
    """Paired t + Wilcoxon for mean(a - b) with fuzzy_dyn as a."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": len(a), "mean_diff": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_t": float("nan"), "p_w": float("nan")}
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(a, b)
    try:
        _, p_t = ttest_rel(a, b)
    except Exception:  # noqa: BLE001
        p_t = float("nan")
    try:
        _, p_w = wilcoxon(a, b)
    except Exception:  # noqa: BLE001
        p_w = float("nan")
    return {"n": len(a), "mean_diff": mean_diff,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_t": float(p_t), "p_w": float(p_w)}


def holm_correction(pvals):
    """Holm step-down adjusted p-values (NaN entries stay NaN)."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    idx = np.where(np.isfinite(pvals))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(pvals[idx])]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out
