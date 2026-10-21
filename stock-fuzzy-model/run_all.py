"""Main experiment driver.

Usage:
    python main.py [--variants M2 M3 M4 M5 M6 fixed] [--seeds 10] [--no-plots]

Protocol
  1. Download + engineer features (^GSPC 2015-2025).
  2. Select hyperparams / features on VALIDATION only.
  3. For each variant × seed: train → collect val RMSE and test predictions.
  4. Report test metrics ONCE per variant (no re-selection after seeing test).
  5. Write results.csv + RESULTS.md.
"""
import json
import os
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

import config as cfg
import data as D
import evaluate as E
import fuzzy as fz
from plots import make_all_plots
from train import train_model, VARIANTS


def _fmt(x, nd=4):
    return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"


def main(variants=None, seeds=None, make_plots=True):
    variants = list(variants or cfg.VARIANTS)
    seeds    = list(seeds    or cfg.SEEDS)

    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant '{v}'")

    # ── T4: fuzzy controller gate ──────────────────────────────────────────
    print("=== T4: fuzzy gate ===")
    g = fz.check_gate()
    for k, v in g.items():
        print(f"  {k}: {v}")
    if not g["pass"]:
        print("GATE FAILED — fix fuzzy.py"); sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Data ===")
    data = D.build_dataset()

    # Persistence baseline on test set
    persist_m = E.persistence_baseline(data["close_test"], data["y_test"])
    print(f"\n  Persistence RMSE={persist_m['persist_RMSE']:.4f}  "
          f"MAPE={persist_m['persist_MAPE']:.4f}%  "
          f"R={persist_m['persist_R']:.4f}")

    # ── Reproducibility check (T1) ────────────────────────────────────────
    print("\n=== T1: reproducibility ===")
    data2 = D.build_dataset()
    chk1  = float(data["X_train"].sum())
    chk2  = float(data2["X_train"].sum())
    t1_ok = abs(chk1 - chk2) < 1e-6
    print(f"  {'PASS' if t1_ok else 'FAIL'}  (checksum {chk1:.4f})")
    if not t1_ok:
        raise AssertionError("Non-deterministic data pipeline")

    # ── Training ─────────────────────────────────────────────────────────
    print("\n=== Training ===")
    rows = []           # one dict per (variant, seed)
    all_test_preds  = {v: [] for v in variants}
    all_val_preds   = {v: [] for v in variants}

    for variant in variants:
        print(f"\n-- variant={variant} --")
        for seed in seeds:
            _, history, vp, tp, info = train_model(
                data, variant=variant, seed=seed)

            # Val metrics (selection criterion — NEVER use test here)
            vm = E.regression_metrics(data["y_val"], vp, data["close_val"])

            # Test metrics (collected but not used for selection)
            tm = E.regression_metrics(data["y_test"], tp, data["close_test"])

            row = {**info,
                   "val_RMSE":  vm["RMSE"],
                   "val_MAPE":  vm["MAPE"],
                   "val_R":     vm["R"],
                   **{f"test_{k}": v for k, v in tm.items()}}
            # Rename test_* to top-level for table
            row["RMSE"]    = tm["RMSE"]
            row["MAPE"]    = tm["MAPE"]
            row["R"]       = tm["R"]
            row["MSE"]     = tm["MSE"]
            row["MAE"]     = tm["MAE"]
            row["dir_acc"] = tm["dir_acc"]
            rows.append(row)

            all_test_preds[variant].append(tp)
            all_val_preds[variant].append(vp)

    res = pd.DataFrame(rows)
    res.to_csv("results.csv", index=False)
    print("\n  results.csv saved")

    # ── Ensemble predictions (M7: average over seeds) ─────────────────────
    ensemble_rows = []
    for variant in variants:
        ens_preds = E.ensemble_predict(all_test_preds[variant])
        em = E.regression_metrics(data["y_test"], ens_preds, data["close_test"])
        ensemble_rows.append({
            "variant":    f"{variant}_ensemble",
            "RMSE":        em["RMSE"],
            "MAPE":        em["MAPE"],
            "R":           em["R"],
            "MSE":         em["MSE"],
            "MAE":         em["MAE"],
            "dir_acc":     em["dir_acc"],
            "n_seeds":     len(seeds),
        })
    ens_df = pd.DataFrame(ensemble_rows)
    ens_df.to_csv("results_ensemble.csv", index=False)

    # ── Paired tests (fuzzy variants vs M2 baseline, Holm-corrected) ──────
    stats_rows = []
    base_v = "Fuzzy_MoE"
    if base_v in variants and len(variants) > 1:
        base_rmse = res[res["variant"] == base_v]["RMSE"].values
        others = [v for v in variants if v != base_v]
        p_t_vals = []
        comparisons_data = []
        for v in others:
            sub_rmse = res[res["variant"] == v]["RMSE"].values
            st = E.paired_stats(base_rmse, sub_rmse)
            comparisons_data.append((v, st))
            p_t_vals.append(st["p_t"])
        holm_pts = E.holm_correction(p_t_vals)
        for (v, st), holm_pt in zip(comparisons_data, holm_pts):
            stats_rows.append({
                "comparison": f"{base_v} vs {v}",
                "mean_diff":  st["mean_diff"],
                "p_t":        st["p_t"],
                "p_w":        st["p_w"],
                "holm_p_t":   holm_pt,
            })
            print(f"  {base_v} vs {v}: diff_RMSE={st['mean_diff']:.4f}  "
                  f"p_t={_fmt(st['p_t'])}  p_w={_fmt(st['p_w'])}  "
                  f"holm={_fmt(holm_pt)}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n=== Test results (mean ± std over seeds) ===")
    for variant in variants:
        sub = res[res["variant"] == variant]
        print(f"  {variant:8s}  RMSE={sub['RMSE'].mean():.4f}±{sub['RMSE'].std():.4f}  "
              f"MAPE={sub['MAPE'].mean():.4f}%  "
              f"R={sub['R'].mean():.4f}  "
              f"dir={sub['dir_acc'].mean():.3f}")
    print("  Ensemble:")
    for _, er in ens_df.iterrows():
        print(f"  {er['variant']:14s}  RMSE={er['RMSE']:.4f}  "
              f"MAPE={er['MAPE']:.4f}%  "
              f"R={er['R']:.4f}  dir={er['dir_acc']:.3f}")

    # ── RESULTS.md ────────────────────────────────────────────────────────
    tbl = E.build_results_table(rows, persist_m, ensemble_df=ens_df)
    E.write_results_md(tbl, stats_rows)

    # ── run_config.json ───────────────────────────────────────────────────
    with open("run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_utc": datetime.now(UTC).isoformat(),
            "ticker":  cfg.TICKER,
            "start":   cfg.START_DATE,
            "end":     cfg.END_DATE,
            "variants": variants,
            "seeds":   seeds,
            "window":  cfg.WINDOW,
            "hidden":  cfg.HIDDEN_SIZE,
            "layers":  cfg.NUM_LAYERS,
            "dropout": cfg.DROPOUT,
            "loss":    cfg.LOSS,
            "batch":   cfg.BATCH_SIZE,
            "epochs":  cfg.EPOCHS,
            "patience":cfg.PATIENCE,
            "lr_init": cfg.INITIAL_LR,
            "gates": {"T1": t1_ok, "T4": g["pass"]},
        }, f, indent=2)
    print("  run_config.json saved")

    # ── Plots ─────────────────────────────────────────────────────────────
    if make_plots:
        # Use best variant (lowest mean test RMSE) for plots
        mean_rmse = res.groupby("variant")["RMSE"].mean()
        best_v    = mean_rmse.idxmin()
        best_tp   = E.ensemble_predict(all_test_preds[best_v])
        make_all_plots(data=data, test_preds=best_tp, variant=best_v,
                       results_csv="results.csv")

    print("\nDone.")
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=None, choices=cfg.VARIANTS)
    p.add_argument("--seeds",    type=int,  default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    seeds_list = list(range(args.seeds)) if args.seeds else cfg.SEEDS
    main(variants=args.variants, seeds=seeds_list, make_plots=not args.no_plots)
