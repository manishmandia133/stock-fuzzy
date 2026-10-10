"""Multi-seed experiment driver: seeds x variants -> results.csv + stats + plots."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import config as cfg
import data as D
import fuzzy as fz
from evaluate import evaluate_run
from plots import make_all_plots
from train import VARIANTS, train_variant

COMPARE_VARIANTS = ["fixed_1e-3", "fixed_mean_fuzzy", "fuzzy_shuffled_vol"]
COMPARE_METRICS = ["test_auc", "best_val_loss", "highvol_loss_var"]


def _paired(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    try:
        _, p_t = ttest_rel(a, b)
    except Exception:  # noqa: BLE001
        p_t = float("nan")
    try:
        _, p_w = wilcoxon(a, b)
    except Exception:  # noqa: BLE001
        p_w = float("nan")
    print(f"  {name}: paired-t p={p_t:.4f} | wilcoxon p={p_w:.4f}")
    return p_t, p_w


def main(seeds=None, variants=None, make_plots=True):
    seeds = seeds or cfg.SEEDS
    variants = variants or VARIANTS

    print("=== fuzzy gate (Mamdani FIS) ===")
    g = fz.check_gate()
    for k, v in g.items():
        print(f"  {k}: {v}")
    print("  NOTE: curve/corr gates are tautological — they verify the "
          "controller shape, not training benefit.")
    if not g["pass"]:
        print("GATE FAILED — fix fuzzy.py before training.")
        sys.exit(1)
    print("  fuzzy gate: PASS")

    print("=== data ===")
    raw = D.batch_download(cfg.UNIVERSE, cfg.START_DATE, cfg.END_DATE)
    engineered = {t: D.engineer_features(df) for t, df in raw.items()}
    engineered = {t: df for t, df in engineered.items() if len(df) > cfg.WINDOW + 20}
    data, kept = D.build_dataset(engineered)
    print(f"kept tickers: {kept}")

    print("=== split stats ===")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        print(f"  {name}: n={len(y)} up_rate={y.mean():.4f} "
              f"majority={max(y.mean(), 1 - y.mean()):.4f}")

    print("=== reproducibility check (data build twice) ===")
    h1 = float(data["X_train"].sum())
    data2, _ = D.build_dataset(engineered)
    h2 = float(data2["X_train"].sum())
    assert h1 == h2, "non-deterministic data pipeline"
    print(f"  deterministic OK (sum={h1:.4f})")

    rows = []
    for seed in seeds:
        for variant in variants:
            model, history, log_df, info = train_variant(
                data["X_train"], data["y_train"], data["vr_train"],
                data["X_val"], data["y_val"], variant, seed)

            if info["batch_vol_std"] <= 0.15:
                print(f"GATE FAILED for {variant} s{seed}: "
                      f"batch vol std={info['batch_vol_std']:.3f} <= 0.15")
                sys.exit(1)
            if variant == "fuzzy" and not (info["vol_lr_corr"] < -0.7):
                print(f"GATE FAILED: corr(vol,lr)={info['vol_lr_corr']:.3f} >= -0.7.")
                sys.exit(1)

            m = evaluate_run(model, history, log_df,
                             data["X_test"], data["y_test"],
                             data["X_train"][:, -1, :], data["y_train"],
                             seed=seed)
            if m.get("leakage_flag"):
                print(f"WARNING: AUC={m['test_auc']:.3f} > 0.60 suspect leakage "
                      f"({variant} s{seed}) — flagged, not dropped.")
            if m.get("const_pred_flag"):
                print(f"NOTE: {variant} s{seed} is a constant predictor "
                      f"(pos_rate={m['pred_pos_rate']:.3f}, "
                      f"pred_std={m['pred_std']:.4f}).")
            rows.append({"seed": seed, "variant": variant, **info, **m})
            if (isinstance(m.get("test_auc"), float) and
                    abs(history["val_loss"][-1] - np.log(2)) < 0.01):
                print(f"NOTE: {variant} s{seed} val_loss ~ ln2 "
                      f"({history['val_loss'][-1]:.4f}); model may not be learning.")

    res = pd.DataFrame(rows)
    res.to_csv("results.csv", index=False)

    print("=== results: mean ± std per variant ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.groupby("variant")[["test_auc", "test_acc", "test_mcc",
                                      "best_val_loss", "train_val_gap",
                                      "pred_pos_rate", "pred_std"]].agg(["mean", "std"]).to_string())

    print("=== test AUC per seed + bootstrap 95% CI ===")
    for _, r in res.iterrows():
        print(f"  {r['variant']} s{int(r['seed'])}: "
              f"AUC={r['test_auc']:.4f} [{r['test_auc_lo']:.4f}, {r['test_auc_hi']:.4f}] "
              f"pos_rate={r['pred_pos_rate']:.3f} pred_std={r['pred_std']:.4f}")

    n_const = int(res["const_pred_flag"].sum())
    print(f"=== constant-predictor summary: {n_const}/{len(res)} runs "
          f"predict a single class at 0.5 threshold ===")

    print("=== paired tests: fuzzy vs controls (p-values UNCORRECTED) ===")
    fz_res = res[res["variant"] == "fuzzy"].set_index("seed").sort_index()
    for other in COMPARE_VARIANTS:
        ot = res[res["variant"] == other].set_index("seed").sort_index()
        print(f"fuzzy vs {other}:")
        for metric in COMPARE_METRICS:
            _paired(fz_res[metric].values, ot[metric].values,
                    f"  {metric}")

    if make_plots:
        make_all_plots()
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
