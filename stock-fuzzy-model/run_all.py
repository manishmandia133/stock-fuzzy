"""Multi-seed experiment driver (PLAN T8): seeds x variants -> results.csv + stats."""
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


def _paired(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        _, p_t = ttest_rel(a, b)
    except Exception:  # noqa: BLE001
        p_t = float("nan")
    try:
        _, p_w = wilcoxon(a, b)
    except Exception:  # noqa: BLE001
        p_w = float("nan")
    print(f"{name}: paired t p={p_t:.4f} | wilcoxon p={p_w:.4f}")
    return p_t, p_w


def main(seeds=None, variants=None, make_plots=True):
    seeds = seeds or cfg.SEEDS
    variants = variants or VARIANTS

    print("=== T4 gate: fuzzy curve ===")
    g = fz.check_gate()
    print(g)
    if not g["pass"]:
        print("T4 GATE FAILED — fix fuzzy.py before training.")
        sys.exit(1)

    print("=== data ===")
    raw = D.batch_download(cfg.UNIVERSE, cfg.START_DATE, cfg.END_DATE)
    engineered = {t: D.engineer_features(df) for t, df in raw.items()}
    engineered = {t: df for t, df in engineered.items() if len(df) > cfg.WINDOW + 20}
    data, kept = D.build_dataset(engineered)
    print(f"kept tickers: {kept}")

    # T1 gate: two runs same seed -> identical results (checked on data build)
    print("=== T1 gate: reproducibility (data build twice, compare hashes) ===")
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

            # T5 gate
            if info["batch_vol_std"] <= 0.15:
                print(f"T5 GATE FAILED for {variant} s{seed}: "
                      f"batch vol std={info['batch_vol_std']:.3f} <= 0.15")
                sys.exit(1)
            # T6 gate (fuzzy variant only)
            if variant == "fuzzy" and not (info["vol_lr_corr"] < -0.7):
                print(f"T6 GATE FAILED: corr(vol,lr)={info['vol_lr_corr']:.3f} >= -0.7. "
                      "STOP — fix T4/T5.")
                sys.exit(1)

            m = evaluate_run(model, history, log_df,
                             data["X_test"], data["y_test"],
                             data["X_train"][:, -1, :], data["y_train"])
            if m.get("leakage_flag"):
                print(f"WARNING T7: AUC={m['test_auc']:.3f} > 0.60 suspect leakage "
                      f"({variant} s{seed}) — flagged, not dropped.")
            rows.append({"seed": seed, "variant": variant, **info, **m})

    res = pd.DataFrame(rows)
    res.to_csv("results.csv", index=False)
    print(res.groupby("variant")[["test_auc", "test_acc", "test_mcc"]].agg(["mean", "std"]))

    # T8 stats: skipped (single variant: fuzzy only, nothing to compare)

    if make_plots:
        make_all_plots()
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
