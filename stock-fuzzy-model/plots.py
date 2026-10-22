"""Plots for the ^GSPC fuzzy regression experiment. Saved to figs/ at 300 dpi."""
import glob
import os
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import fuzzy as fz

plt.switch_backend("Agg")

FIGDIR = "figs"


def _save(fig, name: str):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _variant(results_csv="results.csv") -> str:
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        names = [v for v in df["variant"].unique() if "ensemble" not in v]
        if names:
            return names[0]
    return cfg.VARIANTS[0]


def _hist_files(variant, logdir="logs"):
    return sorted(glob.glob(os.path.join(logdir, f"{variant}_s*_hist.csv")))


def _batch_files(variant, logdir="logs"):
    return sorted(glob.glob(os.path.join(logdir, f"{variant}_s*_batch.csv")))


def _stack_series(files, column):
    series = []
    for path in files:
        df = pd.read_csv(path)
        if column in df.columns:
            series.append(df[column].values)
    if not series:
        return None
    length = max(len(s) for s in series)
    arr = np.full((len(series), length), np.nan)
    for i, s in enumerate(series):
        arr[i, :len(s)] = s
    return arr


# ── 01: True vs Predicted price ─────────────────────────────────────────────

def plot_true_vs_pred(dates, true_price, pred_price, variant="best",
                      split="test"):
    fig, ax = plt.subplots(figsize=(12, 4))
    dates_p = pd.to_datetime(dates)
    ax.plot(dates_p, true_price, label="True Close", lw=1.2)
    ax.plot(dates_p, pred_price, label="Pred Close", lw=1.0, alpha=0.8, ls="--")
    ax.set_title(f"True vs Predicted ^GSPC Close — {split} set ({variant})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (USD)")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"01_true_vs_pred_{variant}.png")


# ── 02: Prediction error over time ──────────────────────────────────────────

def plot_error_over_time(dates, errors, variant="best"):
    fig, ax = plt.subplots(figsize=(12, 3))
    dates_p = pd.to_datetime(dates)
    ax.bar(dates_p, errors, color=np.where(errors >= 0, "steelblue", "tomato"),
           width=1.0, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Prediction Error (True − Pred) over time ({variant})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (USD)")
    fig.tight_layout()
    _save(fig, f"02_error_over_time_{variant}.png")


# ── 03: LR over training epochs ─────────────────────────────────────────────

def plot_lr_over_steps(logdir="logs", variants=None):
    variants = variants or cfg.VARIANTS
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for v in variants:
        arr = _stack_series(_hist_files(v, logdir), "lr_epoch")
        if arr is None:
            continue
        m = np.nanmean(arr, axis=0) * 1e3
        sd = np.nanstd(arr, axis=0) * 1e3
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, label=v)
        ax.fill_between(ep, m - sd, m + sd, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(cfg.INITIAL_LR * 1e3, color="k", ls=":", label="initial LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean LR (×1e-3)")
    ax.set_title("Fuzzy LR over epochs (mean ± std over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "03_lr_over_epochs.png")


# ── 04: Membership functions ────────────────────────────────────────────────

def plot_membership_functions():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    v = fz.V_GRID
    ax[0].plot(v, fz.mu_low(v), label="Low")
    ax[0].plot(v, fz.mu_med(v), label="Med")
    ax[0].plot(v, fz.mu_high(v), label="High")
    ax[0].set_title("Input MFs: vol_rank (Ruspini, sum=1)")
    ax[0].set_xlabel("vol_rank")
    ax[0].set_ylabel("membership")
    ax[0].legend()

    u = fz.OUT_GRID * 1e3
    ax[1].plot(u, fz.SLOW_MF, label=f"Slow (c={cfg.LR_SLOW_C*1e3:.1f}×1e-3)")
    ax[1].plot(u, fz.MED_MF, label=f"Med  (c={cfg.LR_MED_C*1e3:.1f}×1e-3)")
    ax[1].plot(u, fz.FAST_MF, label=f"Fast (c={cfg.LR_FAST_C*1e3:.1f}×1e-3)")
    ax[1].set_title("Output MFs: LR")
    ax[1].set_xlabel("LR (×1e-3)")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "04_membership_functions.png")


# ── 05: Residual distribution ────────────────────────────────────────────────

def plot_residuals(errors, variant="best"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=50, density=True, alpha=0.6, label="Residuals")
    ax.axvline(0, color="k", ls="--", label="zero")
    ax.axvline(np.mean(errors), color="tomato", ls="-",
               label=f"mean={np.mean(errors):.2f}")
    ax.axvline(np.median(errors), color="green", ls=":",
               label=f"median={np.median(errors):.2f}")
    ax.set_title(f"Residual distribution ({variant}, test set)")
    ax.set_xlabel("Error (USD)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, f"05_residuals_{variant}.png")


# ── 06: RMSE boxplot per variant ─────────────────────────────────────────────

def plot_rmse_boxplot(results_csv="results.csv"):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    variants = df["variant"].unique().tolist()
    data_vals = [df.loc[df["variant"] == v, "RMSE"].values for v in variants]
    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        ax.boxplot(data_vals, tick_labels=variants, showmeans=True)
    except TypeError:
        ax.boxplot(data_vals, labels=variants, showmeans=True)
    ax.set_ylabel("Test RMSE (USD)")
    ax.set_title("Test RMSE by variant")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "06_rmse_boxplot.png")


# ── 07: Fuzzy LR controller vs observed batch LR ─────────────────────────────

def plot_lr_controller(variant="Fuzzy_Input", logdir="logs"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fz.V_GRID, fz.LR_CURVE * 1e3, color="k", lw=2,
            label="controller curve")
    xs, ys = [], []
    for path in _batch_files(variant, logdir):
        df = pd.read_csv(path)
        if {"vol", "lr"}.issubset(df.columns):
            xs.append(df["vol"].values)
            ys.append(df["lr"].values * 1e3)
    if xs:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        rng = np.random.default_rng(0)
        take = rng.choice(len(x), size=min(2500, len(x)), replace=False)
        ax.scatter(x[take], y[take], s=6, alpha=0.25, label="batch samples")
    ax.set_xlabel("vol_rank")
    ax.set_ylabel("LR (×1e-3)")
    ax.set_title("Fuzzy controller: vol_rank → learning rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "07_lr_controller.png")


# ── 08: Train / validation loss curves ───────────────────────────────────────

def plot_loss_curves(variant="Fuzzy_Input", logdir="logs"):
    files = _hist_files(variant, logdir)
    if not files:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ls = {"train_loss": "C0", "val_loss": "C1"}
    for col, color in ls.items():
        arr = _stack_series(files, col)
        if arr is None:
            continue
        m = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, color=color, label=col.replace("_", " ").title())
        ax.fill_between(ep, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title(f"{variant} loss curves (mean ± std over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "08_loss_curves.png")


# ── 09: vol_rank distribution by split ───────────────────────────────────────

def plot_vol_rank_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    for split, color in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
        vr = data.get(f"vr_{split}")
        if vr is not None:
            ax.hist(vr, bins=30, density=True, alpha=0.45, color=color,
                    label=split)
    ax.set_xlabel("vol_rank")
    ax.set_ylabel("density")
    ax.set_title("Volatility-regime distribution by split")
    ax.legend()
    fig.tight_layout()
    _save(fig, "09_vol_rank_distribution.png")


# ── 10: Test RMSE per seed ───────────────────────────────────────────────────

def plot_rmse_by_seed(results_csv="results.csv", data=None):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    df = df[~df["variant"].str.contains("ensemble")]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, variant in enumerate(df["variant"].unique()):
        sub = df[df["variant"] == variant].sort_values("seed")
        ax.bar(sub["seed"] + i * 0.35, sub["RMSE"], width=0.35,
               label=variant)
    ax.axhline(df["RMSE"].mean(), color="k", ls="--",
               label=f"mean={df['RMSE'].mean():.2f}")
    if data is not None:
        persist = np.sqrt(np.mean((data["close_test"]
                                   * np.exp(data["y_test"])
                                   - data["close_test"]) ** 2))
        ax.axhline(persist, color="tomato", ls=":", label="persistence")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test RMSE (USD)")
    ax.set_title("Test RMSE per seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "10_rmse_by_seed.png")


# ── 11: Prediction scatter (returns and prices) ──────────────────────────────

def plot_prediction_scatter(data, test_preds, variant="best"):
    import evaluate as E

    y_true = data["y_test"]
    true_price = E.reconstruct_close(data["close_test"], y_true)
    pred_price = E.reconstruct_close(data["close_test"], test_preds)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(y_true * 100, test_preds * 100, s=10, alpha=0.5)
    lim = max(np.abs(y_true).max(), np.abs(test_preds).max()) * 100
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[0].set_xlabel("True next-day return (%)")
    ax[0].set_ylabel("Predicted next-day return (%)")
    ax[0].set_title("Return predictions")
    ax[0].axis("equal")

    ax[1].scatter(true_price, pred_price, s=10, alpha=0.5)
    lo = min(true_price.min(), pred_price.min())
    hi = max(true_price.max(), pred_price.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[1].set_xlabel("True next Close (USD)")
    ax[1].set_ylabel("Predicted next Close (USD)")
    ax[1].set_title("Price predictions")
    fig.suptitle(f"Prediction scatter ({variant}, test set)")
    fig.tight_layout()
    _save(fig, "11_prediction_scatter.png")


# ── 12: Rolling RMSE over test time ──────────────────────────────────────────

def plot_rolling_rmse(data, test_preds, window=20):
    import evaluate as E

    true_price = E.reconstruct_close(data["close_test"], data["y_test"])
    pred_price = E.reconstruct_close(data["close_test"], test_preds)
    dates = pd.to_datetime(data["dates_test"])

    model_err = pd.Series(true_price - pred_price, index=dates)
    persist_err = pd.Series(true_price - data["close_test"], index=dates)
    model_rmse = np.sqrt(model_err.pow(2).rolling(window).mean())
    persist_rmse = np.sqrt(persist_err.pow(2).rolling(window).mean())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(model_rmse.index, model_rmse.values, label="Fuzzy model")
    ax.plot(persist_rmse.index, persist_rmse.values, label="Persistence")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling RMSE ({window}-day, USD)")
    ax.set_title("Rolling test RMSE: fuzzy model vs persistence")
    ax.legend()
    fig.tight_layout()
    _save(fig, "12_rolling_rmse.png")


# ── 13: Fuzzy memberships over the test period ───────────────────────────────

def plot_memberships_over_time(data):
    vr = np.asarray(data["vr_test"], float)
    dates = pd.to_datetime(data["dates_test"])
    low, med, high = fz.mu_low(vr), fz.mu_med(vr), fz.mu_high(vr)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(dates, low, med, high,
                 labels=["Low vol", "Med vol", "High vol"],
                 colors=["C0", "C1", "C2"], alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("membership")
    ax.set_title("Ruspini volatility memberships over the test period")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, "13_memberships_over_time.png")


# ── 14: Error by volatility regime ───────────────────────────────────────────

def plot_error_by_regime(data, test_preds):
    import evaluate as E

    vr = np.asarray(data["vr_test"], float)
    true_price = E.reconstruct_close(data["close_test"], data["y_test"])
    pred_price = E.reconstruct_close(data["close_test"], test_preds)
    persist_price = data["close_test"]

    edges = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001]
    labels = ["Low", "Med", "High"]
    model_rmse, persist_rmse = [], []
    for lo, hi in pairwise(edges):
        mask = (vr >= lo) & (vr < hi)
        if not np.any(mask):
            model_rmse.append(np.nan)
            persist_rmse.append(np.nan)
            continue
        model_rmse.append(np.sqrt(np.mean(
            (true_price[mask] - pred_price[mask]) ** 2)))
        persist_rmse.append(np.sqrt(np.mean(
            (true_price[mask] - persist_price[mask]) ** 2)))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, model_rmse, width=0.4, label="Fuzzy model")
    ax.bar(x + 0.2, persist_rmse, width=0.4, label="Persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Volatility regime")
    ax.set_ylabel("Test RMSE (USD)")
    ax.set_title("Test RMSE by volatility regime")
    ax.legend()
    fig.tight_layout()
    _save(fig, "14_error_by_regime.png")


# ── 15: Batch learning-rate distribution ─────────────────────────────────────

def plot_lr_distribution(variant="Fuzzy_Input", logdir="logs"):
    vals = []
    for path in _batch_files(variant, logdir):
        df = pd.read_csv(path)
        if "lr" in df.columns:
            vals.append(df["lr"].values)
    if not vals:
        return
    vals = np.concatenate(vals) * 1e3
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=50, alpha=0.7, color="C0")
    ax.axvline(np.mean(vals), color="k", ls="--",
               label=f"mean={np.mean(vals):.3f}×1e-3")
    ax.set_xlabel("LR (×1e-3)")
    ax.set_ylabel("batch count")
    ax.set_title("Learning rate selected by the fuzzy controller")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "15_lr_distribution.png")


# ── 16: Price history with train/val/test splits ─────────────────────────────

def plot_price_splits(data):
    fig, ax = plt.subplots(figsize=(12, 4))
    for split, color in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
        dates = data.get(f"dates_{split}")
        close = data.get(f"close_{split}")
        if dates is not None and close is not None:
            ax.plot(pd.to_datetime(dates), close, color=color, label=split)
    for split in ("val", "test"):
        dates = data.get(f"dates_{split}")
        if dates is not None and len(dates):
            ax.axvline(pd.to_datetime(dates[0]), color="k", ls=":", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (USD)")
    ax.set_title("^GSPC Close with chronological train / val / test splits")
    ax.legend()
    fig.tight_layout()
    _save(fig, "16_price_splits.png")


# ── Main entry ───────────────────────────────────────────────────────────────

def make_all_plots(data: dict, test_preds: np.ndarray,
                   variant: str = "best", results_csv: str = "results.csv"):
    """Generate all plots from a data dict and test predictions."""
    import evaluate as E

    true_price = E.reconstruct_close(data["close_test"], data["y_test"])
    pred_price = E.reconstruct_close(data["close_test"], test_preds)
    errors = true_price - pred_price
    dates = data["dates_test"]

    plot_true_vs_pred(dates, true_price, pred_price, variant=variant)
    plot_error_over_time(dates, errors, variant=variant)
    plot_lr_over_steps()
    plot_membership_functions()
    plot_residuals(errors, variant=variant)
    plot_rmse_boxplot(results_csv)
    plot_lr_controller(variant=variant)
    plot_loss_curves(variant=variant)
    plot_vol_rank_distribution(data)
    plot_rmse_by_seed(results_csv, data=data)
    plot_prediction_scatter(data, test_preds, variant=variant)
    plot_rolling_rmse(data, test_preds)
    plot_memberships_over_time(data)
    plot_error_by_regime(data, test_preds)
    plot_lr_distribution(variant=variant)
    plot_price_splits(data)

    print(f"  Figures saved to {FIGDIR}/")


def _load_plots_data(path="logs/plots_data.npz"):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _load_ensemble_preds(variant, logdir="logs"):
    path = os.path.join(logdir, f"{variant}_ensemble_preds.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["test_pred"]


if __name__ == "__main__":
    variant = _variant()
    print(f"Generating figures for {variant}...")
    data = _load_plots_data()
    if data is None:
        import data as D
        data = D.build_dataset()
    test_preds = _load_ensemble_preds(variant)
    if test_preds is None:
        print("  No saved predictions; using persistence (zero return).")
        test_preds = np.zeros_like(data["y_test"])
    make_all_plots(data=data, test_preds=test_preds, variant=variant)
    print("Done generating figures.")
