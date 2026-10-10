"""Plots. All figures -> figs/ as PNG (300 dpi), numbered 01..07 (+ extras)."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib_version = tuple(int(t) for t in matplotlib.__version__.split(".")[:2])

import config as cfg
import fuzzy as fz

FIGDIR = "figs"
LN2 = float(np.log(2))


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_01_membership_input():
    fig, ax = plt.subplots(figsize=(7, 4))
    v = fz.V_GRID
    ax.plot(v, fz.mu_low(v), label="Low")
    ax.plot(v, fz.mu_med(v), label="Medium")
    ax.plot(v, fz.mu_high(v), label="High")
    ax.set_title("Input MFs: vol_rank (Ruspini triangular partition)")
    ax.set_xlabel("vol_rank")
    ax.set_ylabel("membership")
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_membership_input.png")


def plot_02_membership_output():
    fig, ax = plt.subplots(figsize=(7, 4))
    u = fz.OUT_GRID * 1e3
    ax.plot(u, fz.SLOW_MF, label="Slow (c=0.4e-3)")
    ax.plot(u, fz.MED_MF, label="Medium (c=1.0e-3)")
    ax.plot(u, fz.FAST_MF, label="Fast (c=1.6e-3)")
    ax.set_title("Output MFs: learning rate")
    ax.set_xlabel("LR (1e-3)")
    ax.set_ylabel("membership")
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_membership_output.png")


def plot_03_control_curve():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fz.V_GRID, np.array(fz.LR_CURVE) * 1e3)
    ax.set_title("Mamdani control curve (high vol -> lower LR)")
    ax.set_xlabel("vol_rank")
    ax.set_ylabel("LR (1e-3)")
    fig.tight_layout()
    _save(fig, "03_control_curve.png")


def plot_04_worked_example(v0=0.7):
    """Worked Mamdani inference at one vol_rank: firing levels, clipped
    consequents, aggregated output + centroid."""
    a_low, a_med, a_high = (float(fz.mu_low(v0)), float(fz.mu_med(v0)),
                            float(fz.mu_high(v0)))
    imp_fast = np.minimum(a_low, fz.FAST_MF)
    imp_med = np.minimum(a_med, fz.MED_MF)
    imp_slow = np.minimum(a_high, fz.SLOW_MF)
    agg = np.maximum(np.maximum(imp_fast, imp_med), imp_slow)
    lr_star = float((agg * fz.OUT_GRID).sum() / max(agg.sum(), 1e-18))
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    u = fz.OUT_GRID * 1e3
    ax[0].plot(u, fz.SLOW_MF, "--", lw=1, label="Slow")
    ax[0].plot(u, fz.MED_MF, "--", lw=1, label="Medium")
    ax[0].plot(u, fz.FAST_MF, "--", lw=1, label="Fast")
    ax[0].plot(u, imp_slow, label=f"Slow clipped (a={a_high:.2f})")
    ax[0].plot(u, imp_med, label=f"Med clipped (a={a_med:.2f})")
    ax[0].plot(u, imp_fast, label=f"Fast clipped (a={a_low:.2f})")
    ax[0].set_ylabel("membership")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"Worked inference at vol_rank={v0}")
    ax[1].fill_between(u, agg, alpha=0.4, label="aggregated (max)")
    ax[1].axvline(lr_star * 1e3, color="k", linestyle="--",
                  label=f"centroid LR={lr_star*1e3:.3f}e-3")
    ax[1].set_xlabel("LR (1e-3)")
    ax[1].set_ylabel("membership")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "04_worked_example.png")


def plot_05_price_vol_lr(data_csv="data/processed_GSPC.csv"):
    if not os.path.exists(data_csv):
        return
    df = pd.read_csv(data_csv, parse_dates=["Date"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df["Date"], df["Close"])
    ax[0].set_ylabel("Close")
    ax[0].set_title("^GSPC price / volatility / fuzzy LR with split shading")
    ax[1].plot(df["Date"], df["volatility_14d"], color="orange")
    ax[1].set_ylabel("volatility_14d")
    ax[2].plot(df["Date"], np.array(df["fuzzy_lr"]) * 1e3, color="green")
    ax[2].set_ylabel("fuzzy LR (1e-3)")
    for a in ax:
        for s, c in [("train", "green"), ("val", "gold"), ("test", "red")]:
            sub = df[df["split"] == s]
            if len(sub):
                a.axvspan(sub["Date"].iloc[0], sub["Date"].iloc[-1],
                          color=c, alpha=0.07)
    ax[2].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "05_price_vol_lr.png")


def plot_06_loss_curves(logdir="logs"):
    """Train AND val loss per variant (mean +/- std over seeds) + ln2 line."""
    from train import VARIANTS

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for j, split in enumerate(["train_loss", "val_loss"]):
        ax = axes[j]
        for variant in VARIANTS:
            vals = []
            for s in cfg.SEEDS:
                p = os.path.join(logdir, f"{variant}_s{s}_hist.csv")
                if os.path.exists(p):
                    h = pd.read_csv(p)
                    if split in h.columns:
                        vals.append(h[split].values)
            if vals:
                L = max(len(v) for v in vals)
                arr = np.full((len(vals), L), np.nan)
                for i, v in enumerate(vals):
                    arr[i, :len(v)] = v
                m = np.nanmean(arr, axis=0)
                sd = np.nanstd(arr, axis=0)
                ep = np.arange(1, L + 1)
                ax.plot(ep, m, label=variant)
                ax.fill_between(ep, m - sd, m + sd, alpha=0.2)
        ax.axhline(LN2, color="k", linestyle=":", label="ln2 (chance BCE)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{split} (mean ± std over seeds)")
        ax.set_title(f"{'Train' if j == 0 else 'Val'} loss per variant")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "06_loss_curves.png")


def plot_07_auc_boxplot(results_csv="results.csv"):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(9, 4))
    variants = list(df["variant"].unique())
    data = [df.loc[df["variant"] == v, "test_auc"].values for v in variants]
    try:
        ax.boxplot(data, tick_labels=variants)
    except TypeError:
        ax.boxplot(data, labels=variants)  # matplotlib < 3.9 fallback
    ax.set_ylabel("test AUC")
    ax.set_title("AUC per variant (5 seeds each)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "07_auc_boxplot.png")


def plot_lr_vs_vol(logdir="logs", seed=0):
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["vol"], np.array(df["lr"]) * 1e3, s=8, alpha=0.5)
    ax.set_xlabel("batch volatility (mean vol_rank)")
    ax.set_ylabel("LR (1e-3)")
    ax.set_title(f"Fuzzy LR vs volatility (seed {seed})")
    fig.tight_layout()
    _save(fig, f"08_lr_vs_vol_s{seed}.png")


def plot_lr_over_time(logdir="logs", seed=0):
    """LR schedule over optimizer steps (shuffled batch order, NOT time)."""
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["lr"].values * 1e3, label="LR")
    ax1.set_xlabel("optimizer step (shuffled batch order, not time)")
    ax1.set_ylabel("LR (1e-3)")
    ax2 = ax1.twinx()
    ax2.plot(df["vol"].values, color="orange", alpha=0.5, label="vol")
    ax2.set_ylabel("batch vol")
    ax1.set_title(f"LR schedule over optimizer steps (seed {seed})")
    fig.tight_layout()
    _save(fig, f"09_lr_over_step_s{seed}.png")


def make_all_plots():
    plot_01_membership_input()
    plot_02_membership_output()
    plot_03_control_curve()
    plot_04_worked_example()
    plot_05_price_vol_lr()
    plot_06_loss_curves()
    plot_07_auc_boxplot()
    plot_lr_vs_vol()
    plot_lr_over_time()
    print(f"Figures saved to {FIGDIR}/")
