"""Plots (PLAN T9). All figures -> figs/ as PNG (300 dpi)."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import fuzzy as fz

FIGDIR = "figs"


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_membership_and_curve():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    v = fz.V_GRID
    ax[0].plot(v, fz._low, label="Low")
    ax[0].plot(v, fz._med, label="Med")
    ax[0].plot(v, fz._high, label="High")
    ax[0].set_title("Volatility membership functions")
    ax[0].set_xlabel("vol_rank")
    ax[0].legend()
    ax[1].plot(v, fz.LR_CURVE)
    ax[1].set_title("LR curve (high vol -> lower LR)")
    ax[1].set_xlabel("vol_rank")
    ax[1].set_ylabel("LR")
    fig.tight_layout()
    _save(fig, "membership_lr_curve.png")


def plot_lr_vs_vol(logdir="logs", seed=0):
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["vol"], df["lr"], s=8, alpha=0.5)
    ax.set_xlabel("batch volatility (mean vol_rank)")
    ax.set_ylabel("LR")
    ax.set_title(f"Fuzzy LR vs volatility (seed {seed})")
    fig.tight_layout()
    _save(fig, f"lr_vs_vol_s{seed}.png")


def plot_lr_over_time(logdir="logs", seed=0):
    p = os.path.join(logdir, f"fuzzy_s{seed}.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    ep1 = df[df["epoch"] == 1].reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(ep1["lr"].values, label="LR")
    ax1.set_xlabel("batch in epoch 1")
    ax1.set_ylabel("LR")
    ax2 = ax1.twinx()
    ax2.plot(ep1["vol"].values, color="orange", alpha=0.7, label="vol")
    ax2.set_ylabel("vol")
    ax1.set_title(f"LR over time with vol overlay (seed {seed})")
    fig.tight_layout()
    _save(fig, f"lr_over_time_s{seed}.png")


def plot_val_curves(results_csv="results.csv"):
    if not os.path.exists(results_csv):
        return
    # val curves are reconstructed from logs if present; fallback: skip
    import glob
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in cfg_SEEDS_VARIANTS()[0]:
        vals = []
        for s in cfg.SEEDS:
            p = os.path.join("logs", f"{variant}_s{s}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                vals.append(df.groupby("epoch")["loss"].mean().values)
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
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (mean ± std across seeds)")
    ax.set_title("Loss curves per variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "loss_curves.png")


def plot_auc_boxplot(results_csv="results.csv"):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(9, 4))
    variants = df["variant"].unique()
    ax.boxplot([df.loc[df["variant"] == v, "test_auc"].values for v in variants],
               tick_labels=variants)
    ax.set_ylabel("test AUC")
    ax.set_title("AUC per variant")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "auc_boxplot.png")


def cfg_SEEDS_VARIANTS():
    from train import VARIANTS
    return VARIANTS, cfg.SEEDS


def make_all_plots():
    plot_membership_and_curve()
    plot_lr_vs_vol()
    plot_lr_over_time()
    plot_val_curves()
    plot_auc_boxplot()
    print(f"Figures saved to {FIGDIR}/")
