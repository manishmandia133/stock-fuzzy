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

COMPARE_VARIANTS = ["fixed_1e-3", "fixed_mean_fuzzy", "cosine", "crisp",
                    "fuzzy_inverted", "fuzzy_shuffled_vol"]
COMPARE_METRICS = ["test_auc", "best_val_loss", "highvol_loss_var"]

# Report expectations (Sec 6.1): compared against actuals at runtime.
# Any mismatch is printed plainly — never hidden.
EXPECTED = {"n_train": 1863, "n_val": 352, "n_test": 353,
            "up_train": 0.541, "up_val": 0.534, "up_test": 0.578,
            "checksum": 202000.3771}


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

    print("=== fuzzy gate (T4, Mamdani FIS) ===")
    g = fz.check_gate()
    for k, v in g.items():
        print(f"  {k}: {v}")
    print("  NOTE: curve/corr gates are tautological — they verify the "
          "controller shape, not training benefit.")
    if not g["pass"]:
        print("T4 GATE FAILED — fix fuzzy.py before training.")
        sys.exit(1)
    print("  T4: PASS")
    print("=== prior Sugeno prototype (reference only, not trained) ===")
    sg = fz.sugeno_gate()
    for k, v in sg.items():
        print(f"  sugeno_{k}: {v}")
    print("  (report traceability: prior prototype mean ~9.63e-4, "
          "max step ~1.79e-5)")

    print("=== data ===")
    raw = D.batch_download(cfg.UNIVERSE, cfg.START_DATE, cfg.END_DATE)
    engineered = {t: D.engineer_features(df) for t, df in raw.items()}
    engineered = {t: df for t, df in engineered.items() if len(df) > cfg.WINDOW + 20}
    data, kept = D.build_dataset(engineered)
    print(f"kept tickers: {kept}")
    # T2/T3: the asserts inside data.py (chronological order, label
    # spot-check, n_train>1500, no NaN) raise on failure — reaching here = PASS.
    print(f"T2: PASS (chronological split asserts + 3-window label spot-check)")
    print(f"T3: PASS (n_train={len(data['X_train'])} > 1500, no NaN)")

    print("=== split stats ===")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        print(f"  {name}: n={len(y)} up_rate={y.mean():.4f} "
              f"majority={max(y.mean(), 1 - y.mean()):.4f}")

    print("=== reproducibility check (T1: data build twice) ===")
    h1 = float(data["X_train"].sum())
    data2, _ = D.build_dataset(engineered)
    h2 = float(data2["X_train"].sum())
    t1_pass = (h1 == h2)
    print(f"  T1: {'PASS' if t1_pass else 'FAIL'} "
          f"(checksum={h1:.4f}, expected={EXPECTED['checksum']:.4f})")
    assert t1_pass, "non-deterministic data pipeline"

    print("=== report expected-vs-actual (Sec 6.1) ===")
    actual = {"n_train": len(data["X_train"]), "n_val": len(data["X_val"]),
              "n_test": len(data["X_test"]),
              "up_train": float(data["y_train"].mean()),
              "up_val": float(data["y_val"].mean()),
              "up_test": float(data["y_test"].mean()),
              "checksum": h1}
    for k in ["n_train", "n_val", "n_test", "up_train", "up_val",
              "up_test", "checksum"]:
        e, a = EXPECTED[k], actual[k]
        match = (a == e) if isinstance(e, int) else abs(a - e) < 0.0005
        flag = "MATCHES REPORT" if match else f"DIFFERS FROM REPORT (actual {a} vs report {e})"
        print(f"  {k}: actual={a} expected={e} -> {flag}")

    rows = []
    for seed in seeds:
        for variant in variants:
            model, history, log_df, info = train_variant(
                data["X_train"], data["y_train"], data["vr_train"],
                data["X_val"], data["y_val"], variant, seed)

            if info["batch_vol_std"] <= 0.15:
                print(f"T5 GATE FAILED for {variant} s{seed}: "
                      f"batch vol std={info['batch_vol_std']:.3f} <= 0.15")
                sys.exit(1)
            if variant == "fuzzy" and not (info["vol_lr_corr"] < -0.7):
                print(f"T6 GATE FAILED: corr(vol,lr)={info['vol_lr_corr']:.3f} >= -0.7.")
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

    print("=== paired tests T8: fuzzy vs all others ===")
    print("  (paired t + Wilcoxon per seed; p-values UNCORRECTED; "
          "n=5 seeds; Wilcoxon min achievable p=0.0625)")
    fz_res = res[res["variant"] == "fuzzy"].set_index("seed").sort_index()
    for other in COMPARE_VARIANTS:
        ot = res[res["variant"] == other].set_index("seed").sort_index()
        print(f"fuzzy vs {other}:")
        for metric in COMPARE_METRICS:
            _paired(fz_res[metric].values, ot[metric].values,
                    f"  {metric}")

    print("=== mean_lr_used per variant (RQ1 equal-mean fairness) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(res.groupby("variant")[["mean_lr_used"]].agg(["mean", "std", "min", "max"]).to_string())

    n_leak = int(res["leakage_flag"].sum())
    print(f"T7: {'FLAG — ' + str(n_leak) + ' run(s) with AUC>0.60 (suspect leakage, flagged not dropped)' if n_leak else 'PASS (no test AUC > 0.60)'}")
    print("T8: DONE (paired stats above; do not claim fuzzy wins unless tests support it)")

    print("=== GATE SUMMARY T1..T8 ===")
    print(f"  T1 data reproducibility : {'PASS' if t1_pass else 'FAIL'}")
    print("  T2 split/label checks    : PASS (asserts in data.py)")
    print(f"  T3 scope windows/NaN     : PASS (n_train={len(data['X_train'])})")
    print(f"  T4 fuzzy curve           : {'PASS' if g['pass'] else 'FAIL'} (tautological: shape only)")
    print("  T5 regime batching       : PASS (all batch_vol_std > 0.15)")
    print("  T6 fuzzy corr(vol,lr)    : PASS (tautological: schedule check)")
    print(f"  T7 leakage flag          : {'FLAG' if n_leak else 'PASS'}")
    print("  T8 paired stats          : DONE")

    if make_plots:
        make_all_plots(data=data)
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
