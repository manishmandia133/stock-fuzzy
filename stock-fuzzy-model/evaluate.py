"""Evaluation (PLAN T7): AUC, acc, MCC, majority baseline + stability metrics."""
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
        return model(torch.tensor(X, dtype=torch.float32).to(cfg.DEVICE)).cpu().numpy()


def stability_metrics(history, log_df, vr_train=None):
    val = np.array(history["val_loss"])
    roll = pd_roll(val)
    spikes = int(((val[1:] - roll[:-1]) > val.std()).sum()) if len(val) > 2 else 0
    best = int(np.argmin(val)) + 1
    gap = float(history["train_loss"][best - 1] - val[best - 1])
    hv = float(log_df.loc[log_df["vol"] > 0.7, "loss"].var()) if "vol" in log_df else float("nan")
    return {"spikes": spikes, "epochs_to_best": best,
            "train_val_gap": gap, "highvol_loss_var": hv}


def pd_roll(val):
    s = pd.Series(val)
    return s.rolling(3, min_periods=1).mean().values


def evaluate_run(model, history, log_df, X_test, y_test, X_train_last=None, y_train=None):
    proba = predict_proba(model, X_test)
    pred = (proba > 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    out = {"test_auc": auc,
           "test_acc": float(accuracy_score(y_test, pred)),
           "test_mcc": float(matthews_corrcoef(y_test, pred)),
           "majority_acc": float(max(y_test.mean(), 1 - y_test.mean())),
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
