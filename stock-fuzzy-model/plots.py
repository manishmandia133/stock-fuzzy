"""Plots for the ^GSPC regression experiment. Saved to figs/ at 300 dpi."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import fuzzy as fz

FIGDIR = "figs"


def _save(fig, name: str):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── 01: True vs Predicted price ─────────────────────────────────────────────

def plot_true_vs_pred(dates, true_price, pred_price, variant="best",
                      split="test"):
    """Line plot of true vs predicted Close on the test set."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dates_p = pd.to_datetime(dates)
    ax.plot(dates_p, true_price, label="True Close",  lw=1.2)
    ax.plot(dates_p, pred_price, label="Pred Close",  lw=1.0, alpha=0.8, ls="--")
    ax.set_title(f"True vs Predicted ^GSPC Close — {split} set ({variant})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (USD)")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"01_true_vs_pred_{variant}.png")


# ── 02: Prediction error over time ──────────────────────────────────────────

def plot_error_over_time(dates, errors, variant="best"):
    """Bar-style error over time (True − Predicted in price units)."""
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


# ── 03: LR over training steps ──────────────────────────────────────────────

def plot_lr_over_steps(logdir="logs", variants=None):
    """Mean LR per epoch for each variant (from saved batch logs)."""
    variants = variants or cfg.VARIANTS
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for v in variants:
        # Collect all seeds' epoch-mean LR
        dfs = []
        for s in range(30):
            p = os.path.join(logdir, f"{v}_s{s}_hist.csv")
            if os.path.exists(p):
                h = pd.read_csv(p)
                if "lr_epoch" in h.columns:
                    dfs.append(h["lr_epoch"].values)
        if not dfs:
            continue
        L   = max(len(d) for d in dfs)
        arr = np.full((len(dfs), L), np.nan)
        for i, d in enumerate(dfs):
            arr[i, :len(d)] = d
        m  = np.nanmean(arr, axis=0) * 1e3
        sd = np.nanstd(arr, axis=0) * 1e3
        ep = np.arange(1, L + 1)
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


# ── 04: Membership functions ─────────────────────────────────────────────────

def plot_membership_functions():
    """M2 input (vol_rank) and output (LR) membership functions."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    v = fz.V_GRID
    ax[0].plot(v, fz.mu_low(v),  label="Low")
    ax[0].plot(v, fz.mu_med(v),  label="Med")
    ax[0].plot(v, fz.mu_high(v), label="High")
    ax[0].set_title("Input MFs: vol_rank (Ruspini, sum=1)")
    ax[0].set_xlabel("vol_rank")
    ax[0].set_ylabel("membership")
    ax[0].legend()

    u = fz.OUT_GRID * 1e3
    ax[1].plot(u, fz.SLOW_MF, label=f"Slow (c={cfg.LR_SLOW_C*1e3:.1f}×1e-3)")
    ax[1].plot(u, fz.MED_MF,  label=f"Med  (c={cfg.LR_MED_C*1e3:.1f}×1e-3)")
    ax[1].plot(u, fz.FAST_MF, label=f"Fast (c={cfg.LR_FAST_C*1e3:.1f}×1e-3)")
    ax[1].set_title("Output MFs: LR")
    ax[1].set_xlabel("LR (×1e-3)")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "04_membership_functions.png")


# ── 05: Residual distribution ────────────────────────────────────────────────

def plot_residuals(errors, variant="best"):
    """Histogram + KDE of prediction errors."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=50, density=True, alpha=0.6, label="Residuals")
    ax.axvline(0,                    color="k",      ls="--", label="zero")
    ax.axvline(np.mean(errors),      color="tomato", ls="-",  label=f"mean={np.mean(errors):.2f}")
    ax.axvline(np.median(errors),    color="green",  ls=":",  label=f"median={np.median(errors):.2f}")
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


# ── Main entry ───────────────────────────────────────────────────────────────

def make_all_plots(data: dict, test_preds: np.ndarray,
                   variant: str = "best", results_csv: str = "results.csv"):
    """Generate all plots from data dict and test predictions."""
    import evaluate as E

    true_price = E.reconstruct_close(data["close_test"], data["y_test"])
    pred_price = E.reconstruct_close(data["close_test"], test_preds)
    errors     = true_price - pred_price
    dates      = data["dates_test"]

    plot_true_vs_pred(dates, true_price, pred_price, variant=variant)
    plot_error_over_time(dates, errors, variant=variant)
    plot_lr_over_steps()
    plot_membership_functions()
    plot_residuals(errors, variant=variant)
    plot_rmse_boxplot(results_csv)

    print(f"  Figures saved to {FIGDIR}/")


if __name__ == "__main__":
    import data as D
    print("Generating all figures...")
    d = D.build_dataset()
    # Baseline or best prediction
    if os.path.exists("results.csv"):
        res = pd.read_csv("results.csv")
        best_v = res.groupby("variant")["RMSE"].mean().idxmin()
        print(f"Using best variant from results.csv: {best_v}")
    else:
        best_v = "persistence"
    # Persistence prediction is log_return = 0
    zeros = np.zeros_like(d["y_test"])
    make_all_plots(d, zeros, variant=best_v)
    print("Done generating figures.")
