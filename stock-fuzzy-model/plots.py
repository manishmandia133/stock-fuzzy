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


def plot_01_membership_functions():
    """Two input MF sets (vol_rank, loss_trend) + output LR MFs."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    v = fz.V_GRID
    ax[0].plot(v, fz.mu_low(v), label="Low")
    ax[0].plot(v, fz.mu_med(v), label="Medium")
    ax[0].plot(v, fz.mu_high(v), label="High")
    ax[0].set_title("Input MFs: vol_rank (Ruspini, sum=1)")
    ax[0].set_xlabel("vol_rank")
    ax[0].set_ylabel("membership")
    ax[0].legend()
    t = fz.T_GRID
    ax[1].plot(t, fz.mu_falling(t), label="Falling")
    ax[1].plot(t, fz.mu_plateau(t), label="Plateau")
    ax[1].plot(t, fz.mu_rising(t), label="Rising")
    ax[1].set_title("Input MFs: validation loss_trend")
    ax[1].set_xlabel("loss_trend")
    ax[1].legend()
    u = fz.OUT_GRID * 1e3
    ax[2].plot(u, fz.SLOW_MF, label="Slow (c=0.4e-3)")
    ax[2].plot(u, fz.MED_MF, label="Medium (c=1.0e-3)")
    ax[2].plot(u, fz.FAST_MF, label="Fast (c=1.6e-3)")
    ax[2].set_title("Output MFs: learning rate")
    ax[2].set_xlabel("LR (1e-3)")
    ax[2].legend()
    fig.tight_layout()
    _save(fig, "01_membership_functions.png")


def plot_02_lr_surface():
    """LR surface heatmap + slices at falling/plateau/rising trend."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    mesh = ax[0].pcolormesh(fz.V_GRID, fz.T_GRID,
                            fz.LR_SURFACE.T * 1e3, shading="auto",
                            cmap="viridis")
    fig.colorbar(mesh, ax=ax[0], label="LR (1e-3)")
    ax[0].set_title("Mamdani LR surface: (vol_rank, loss_trend)")
    ax[0].set_xlabel("vol_rank")
    ax[0].set_ylabel("loss_trend")
    for trend, style in [(-1.0, ":"), (0.0, "-"), (1.0, "--")]:
        ax[1].plot(fz.V_GRID, fz.LR_SURFACE[:, np.argmin(np.abs(fz.T_GRID - trend))] * 1e3,
                   style, label=f"trend={trend:+.0f}")
    ax[1].set_title("Surface slices (non-increasing in vol_rank)")
    ax[1].set_xlabel("vol_rank")
    ax[1].set_ylabel("LR (1e-3)")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "02_lr_surface_heatmap.png")


def plot_03_price_vol_lr(data_csv="data/processed_GSPC.csv"):
    if not os.path.exists(data_csv):
        return
    df = pd.read_csv(data_csv, parse_dates=["Date"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df["Date"], df["Close"])
    ax[0].set_ylabel("Close")
    ax[0].set_title("^GSPC price / volatility / fuzzy LR with split shading")
    ax[1].plot(df["Date"], df["volatility_14d"], color="orange")
    ax[1].set_ylabel("volatility_14d")
    lr_ref = np.array([fz.fuzzy_lr_vol_only(v) for v in df["vol_rank"].values])
    ax[2].plot(df["Date"], lr_ref * 1e3, color="green")
    ax[2].set_ylabel("fuzzy LR vol-only (1e-3)")
    for a in ax:
        for s, c in [("train", "green"), ("val", "gold"), ("test", "red")]:
            sub = df[df["split"] == s]
            if len(sub):
                a.axvspan(sub["Date"].iloc[0], sub["Date"].iloc[-1],
                          color=c, alpha=0.07)
    ax[2].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "03_price_vol_lr.png")


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
    from train import VARIANTS

    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(9, 4))
    variants = [v for v in VARIANTS if v in set(df["variant"])]
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


def plot_04_lr_vs_vol(logdir="logs", seed=0):
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "trend" in df.columns:
        sc = ax.scatter(df["vol"], np.array(df["lr"]) * 1e3, c=df["trend"],
                        cmap="coolwarm", vmin=-1, vmax=1, s=8, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="validation loss_trend")
    else:
        ax.scatter(df["vol"], np.array(df["lr"]) * 1e3, s=8, alpha=0.5)
    ax.set_xlabel("batch volatility (mean vol_rank)")
    ax.set_ylabel("LR (1e-3)")
    ax.set_title(f"Fuzzy LR vs volatility (color=trend, seed {seed})")
    fig.tight_layout()
    _save(fig, f"04_lr_vs_vol_s{seed}.png")


def plot_05_lr_over_steps(logdir="logs", seed=0):
    """LR, volatility, and trend over optimizer steps (shuffled batch order)."""
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["lr"].values * 1e3, label="LR")
    ax1.set_ylabel("LR (1e-3)")
    ax1.legend(loc="upper right")
    ax1.set_title(f"LR schedule over optimizer steps (seed {seed})")
    ax2.plot(df["vol"].values, color="orange", alpha=0.6, label="vol_rank")
    if "trend" in df.columns:
        ax2.plot(df["trend"].values, color="purple", alpha=0.8,
                 label="loss_trend")
    ax2.set_xlabel("optimizer step (shuffled batch order, not time)")
    ax2.set_ylabel("input value")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, f"05_lr_over_step_s{seed}.png")


def plot_08_split_balance(data=None, data_csv="data/processed_GSPC.csv"):
    """Report Fig 8: per-split window counts + up-day class balance."""
    if data is None:
        if not os.path.exists(data_csv):
            return
        df = pd.read_csv(data_csv)
        counts, ups = {}, {}
        for s in ["train", "val", "test"]:
            sub = df[df["split"] == s]
            counts[s] = len(sub)
            ups[s] = float(sub["target_direction"].mean()) if len(sub) else float("nan")
    else:
        counts = {k: len(data[f"y_{k}"]) for k in ["train", "val", "test"]}
        ups = {k: float(data[f"y_{k}"].mean()) for k in ["train", "val", "test"]}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = ["train", "val", "test"]
    ax[0].bar(names, [counts[k] for k in names], color=["green", "gold", "red"])
    for k, n in enumerate(names):
        ax[0].text(k, counts[n], str(counts[n]), ha="center", va="bottom")
    ax[0].set_title("Window counts per split")
    ax[0].set_ylabel("n windows")
    ax[1].bar(names, [ups[k] for k in names], color=["green", "gold", "red"])
    ax[1].axhline(0.5, color="k", linestyle=":", label="balanced")
    for k, n in enumerate(names):
        ax[1].text(k, ups[n], f"{ups[n]:.3f}", ha="center", va="bottom")
    ax[1].set_title("Up-day rate per split (majority baseline)")
    ax[1].set_ylabel("up_rate")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, "08_split_balance.png")


def plot_09_gate_margins(n_train=None, data_csv="data/processed_GSPC.csv"):
    """Report Fig 9: T3/T4 measured values vs their gate thresholds."""
    if n_train is None:
        if not os.path.exists(data_csv):
            return
        df = pd.read_csv(data_csv)
        n_train = int((df["split"] == "train").sum())
    g = fz.check_gate()
    pos_steps = np.diff(fz.LR_SURFACE, axis=0)
    max_pos = float(max(pos_steps.max(), 0.0))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(["n_train"], [n_train], color="steelblue")
    ax[0].axhline(1500, color="red", linestyle="--", label="T3 threshold 1500")
    ax[0].set_title(f"T3: n_train={n_train}")
    ax[0].legend()
    ax[1].bar(["mean LR"], [g["mean"] * 1e3], color="steelblue")
    ax[1].axhspan(0.75, 1.25, color="green", alpha=0.2,
                  label="T4 band 1e-3 ±25%")
    ax[1].set_title(f"T4: mean={g['mean']:.2e}")
    ax[1].set_ylabel("LR (1e-3)")
    ax[1].legend()
    ax[2].bar(["max positive step"], [max_pos], color="steelblue")
    ax[2].axhline(g["mono_tol"], color="red", linestyle="--",
                  label=f"T4 tolerance={g['mono_tol']:.2e}")
    ax[2].set_title(f"T4: max positive step={max_pos:.2e}")
    ax[2].legend()
    fig.tight_layout()
    _save(fig, "09_gate_margins.png")


def make_all_plots(data=None):
    plot_01_membership_functions()
    plot_02_lr_surface()
    plot_03_price_vol_lr()
    plot_04_lr_vs_vol()
    plot_05_lr_over_steps()
    plot_06_loss_curves()
    plot_07_auc_boxplot()
    n_train = len(data["X_train"]) if data is not None else None
    plot_08_split_balance(data=data)
    plot_09_gate_margins(n_train=n_train)
    print(f"Figures saved to {FIGDIR}/")
