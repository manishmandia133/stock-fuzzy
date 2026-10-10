"""Evaluation: AUC, acc, MCC, majority baseline + stability + bootstrap CI."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

import config as cfg


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        device = cfg.resolve_device()
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()


def stability_metrics(history, log_df, vr_train=None):
    val = np.array(history["val_loss"])
    train = np.array(history["train_loss"])
    roll = pd_roll(val)
    spikes = int(((val[1:] - roll[:-1]) > val.std()).sum()) if len(val) > 2 else 0
    best = int(np.argmin(val)) + 1
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
        except Exception:  # noqa: BLE001
            continue
    aucs = np.array(aucs)
    if len(aucs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_run(model, history, log_df, X_test, y_test,
                 X_train_last=None, y_train=None, seed=0):
    proba = predict_proba(model, X_test)
    pred = (proba > 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    lo, hi = bootstrap_auc_ci(y_test, proba, seed=seed)
    pos_rate = float(pred.mean())
    pstd = float(proba.std())
    # Constant-predictor flag: model outputs a single class at 0.5 threshold
    const_flag = bool(pos_rate == 0.0 or pos_rate == 1.0)
    out = {"test_auc": auc,
           "test_auc_lo": lo,
           "test_auc_hi": hi,
           "test_acc": float(accuracy_score(y_test, pred)),
           "test_mcc": float(matthews_corrcoef(y_test, pred)),
           "majority_acc": float(max(y_test.mean(), 1 - y_test.mean())),
           "pred_pos_rate": pos_rate,
           "pred_std": pstd,
           "const_pred_flag": const_flag,
           "best_epoch": int(np.argmin(history["val_loss"])) + 1,
           "best_val_loss": float(np.min(history["val_loss"]))}
    out.update(stability_metrics(history, log_df))

    # Gate: AUC > 0.60 suspect leakage (flag, do not silently report)
    out["leakage_flag"] = bool(out["test_auc"] > 0.60)

    # Gate: logistic regression on flattened last-day features ~ 0.50 AUC
    if X_train_last is not None and y_train is not None:
        lr = LogisticRegression(max_iter=1000).fit(X_train_last, y_train)
        out["logreg_auc"] = float(roc_auc_score(
            y_test, lr.predict_proba(X_test[:, -1, :])[:, 1]))
    return out
