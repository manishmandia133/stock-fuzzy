"""Evaluation: AUC, acc, MCC, balanced acc, majority baseline + bootstrap CI."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

import config as cfg


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


def stability_metrics(history, log_df, best_epoch=None, vr_train=None):
    val = np.array(history["val_loss"])
    train = np.array(history["train_loss"])
    roll = pd_roll(val)
    spikes = int(((val[1:] - roll[:-1]) > val.std()).sum()) if len(val) > 2 else 0
    if best_epoch is None:
        best = int(np.argmin(val)) + 1
    else:
        best = int(best_epoch)
    gap = float(val[best - 1] - train[best - 1])  # val_loss - train_loss
    hv = float(log_df.loc[log_df["vol"] > 0.7, "loss"].var()) if "vol" in log_df else float("nan")
    return {"spikes": spikes, "epochs_to_best": best,
            "train_val_gap": gap, "highvol_loss_var": hv}


def pd_roll(val):
    s = pd.Series(val)
    return s.rolling(3, min_periods=1).mean().values


def bootstrap_auc_ci(y_true, proba, n_boot=2000, seed=0):
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
                 X_train_last=None, y_train=None, seed=0):
    proba_val = predict_proba(model, X_val)
    proba = predict_proba(model, X_test)

    thr, thr_bal_acc = select_threshold(y_val, proba_val)
    pred = (proba > thr).astype(int)
    m_thr = _classification_metrics(y_test, pred)
    m_05 = _classification_metrics(y_test, (proba > 0.5).astype(int))

    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    lo, hi = bootstrap_auc_ci(y_test, proba, seed=seed)

    val_auc = np.asarray(history["val_auc"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    if np.any(np.isfinite(val_auc)):
        best_epoch = int(np.nanargmax(val_auc)) + 1
    else:
        best_epoch = int(np.argmin(val_loss)) + 1

    out = {"test_auc": auc,
           "test_auc_lo": lo,
           "test_auc_hi": hi,
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
