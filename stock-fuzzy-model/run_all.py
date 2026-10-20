"""Multi-seed experiment driver: 10 seeds x 2 methods -> results.csv + stats + plots."""
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import config as cfg
import data as D
import fuzzy as fz
from evaluate import evaluate_run
from plots import make_all_plots
from train import VARIANTS, train_variant

BASELINE = "fixed_1e-3"
PROPOSED = "fuzzy"
COMPARE_METRICS = ["test_auc", "test_mcc", "test_acc", "best_val_loss",
                   "highvol_loss_var", "epochs_to_best"]

# Report expectations (Sec 6.1): compared against actuals at runtime.
# Any mismatch is printed plainly — never hidden.
EXPECTED = {"n_train": 1900, "n_val": 391, "n_test": 392,
            "up_train": 0.5400, "up_val": 0.5269, "up_test": 0.5791,
            "checksum": -3942.92227483448}


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

    print("=== data ===")
    raw = D.batch_download(cfg.UNIVERSE, cfg.START_DATE, cfg.END_DATE)
    engineered = {t: D.engineer_features(df) for t, df in raw.items()}
    engineered = {t: df for t, df in engineered.items() if len(df) > cfg.WINDOW + 20}
    data, kept = D.build_dataset(engineered)
    print(f"kept tickers: {kept}")
    # T2/T3: the asserts inside data.py (chronological order, label
    # spot-check, n_train>1500, no NaN) raise on failure — reaching here = PASS.
    print("T2: PASS (chronological split asserts + 3-window label spot-check)")
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

            m = evaluate_run(
                model, history, log_df,
                data["X_val"], data["y_val"],
                data["X_test"], data["y_test"],
                data["X_train"][:, -1, :], data["y_train"],
                seed=seed)
            if m.get("leakage_flag"):
                print(f"WARNING: AUC={m['test_auc']:.3f} > 0.60 suspect leakage "
                      f"({variant} s{seed}) — flagged, not dropped.")
            if m.get("const_pred_flag"):
                print(f"NOTE: {variant} s{seed} is a constant predictor at "
                      f"thr={m['threshold_val']:.3f} "
                      f"(pos_rate={m['pred_pos_rate']:.3f}, "
                      f"pred_std={m['pred_std']:.4f}).")
            if (isinstance(m.get("test_auc"), float) and
                    abs(history["val_loss"][-1] - np.log(2)) < 0.01):
                print(f"NOTE: {variant} s{seed} final val_loss ~ ln2 "
                      f"({history['val_loss'][-1]:.4f}); model may not be learning.")
            rows.append({"seed": seed, "variant": variant, **info, **m})

    res = pd.DataFrame(rows)

    # ---- experimental-integrity checks -----------------------------------
    assert set(res["variant"]) == set(VARIANTS) == {BASELINE, PROPOSED}, \
        f"unexpected variants in results: {sorted(set(res['variant']))}"
    for seed in seeds:
        seed_rows = res[res["seed"] == seed]
        assert set(seed_rows["variant"]) == {BASELINE, PROPOSED}, \
            f"seed {seed} does not contain exactly both methods"
        assert seed_rows["n_train"].nunique() == 1, \
            f"seed {seed}: train size differs across methods"
        assert seed_rows["n_val"].nunique() == 1, \
            f"seed {seed}: val size differs across methods"
    # Baseline LR must be exactly constant; fuzzy must vary and stay in bounds.
    base = res[res["variant"] == BASELINE]
    assert bool((base["lr_min_used"] == cfg.FIXED_LR).all()), \
        "fixed_1e-3 did not use exactly cfg.FIXED_LR"
    assert bool((base["lr_max_used"] == cfg.FIXED_LR).all()), \
        "fixed_1e-3 changed its learning rate"
    assert bool((base["lr_std_used"] == 0.0).all()), \
        "fixed_1e-3 learning rate has non-zero variance"
    fzr = res[res["variant"] == PROPOSED]
    assert bool((fzr["lr_min_used"] >= cfg.LR_MIN).all()), "fuzzy LR below bounds"
    assert bool((fzr["lr_max_used"] <= cfg.LR_MAX).all()), "fuzzy LR above bounds"
    assert bool((fzr["lr_std_used"] > 0.0).all()), "fuzzy LR never changed"
    assert bool((fzr["vol_lr_corr"] < -0.7).all()), \
        "fuzzy LR did not respond inversely to batch volatility (T6)"

    res.to_csv("results.csv", index=False)
    with open("run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_utc": datetime.now(UTC).isoformat(),
            "methods": VARIANTS,
            "seeds": list(seeds),
            "config": {
                "universe": cfg.UNIVERSE, "start_date": cfg.START_DATE,
                "end_date": cfg.END_DATE, "window": cfg.WINDOW,
                "batch_size": cfg.BATCH_SIZE, "max_epochs": cfg.EPOCHS,
                "patience": cfg.PATIENCE, "min_epochs": cfg.MIN_EPOCHS,
                "grad_clip": cfg.GRAD_CLIP,
                "weight_decay": cfg.WEIGHT_DECAY,
                "hidden_size": cfg.HIDDEN_SIZE, "num_layers": cfg.NUM_LAYERS,
                "dropout": cfg.DROPOUT, "clip": cfg.CLIP,
                "threshold_grid": [cfg.THRESHOLD_LOW, cfg.THRESHOLD_HIGH,
                                   cfg.THRESHOLD_STEP],
                "train_frac": cfg.TRAIN_FRAC, "val_frac": cfg.VAL_FRAC,
                "features": cfg.FEATURE_COLS, "target": cfg.TARGET_COL,
                "fixed_lr": cfg.FIXED_LR, "fuzzy_lr_min": cfg.LR_MIN,
                "fuzzy_lr_max": cfg.LR_MAX,
                "fuzzy_centroids": [cfg.LR_SLOW_C, cfg.LR_MED_C,
                                    cfg.LR_FAST_C],
            },
        }, f, indent=2)

    print("=== results: mean ± std per method ===")
    with pd.option_context("display.width", 220, "display.max_columns", 24):
        print(res.groupby("variant")[["test_auc", "test_acc", "test_mcc",
                                      "test_bal_acc", "threshold_val",
                                      "best_val_loss", "train_val_gap",
                                      "pred_pos_rate", "pred_std",
                                      "const_pred_flag"]].agg(["mean", "std"]).to_string())

    print("=== test AUC per seed + bootstrap 95% CI ===")
    for _, r in res.iterrows():
        print(f"  {r['variant']} s{int(r['seed'])}: "
              f"AUC={r['test_auc']:.4f} [{r['test_auc_lo']:.4f}, {r['test_auc_hi']:.4f}] "
              f"thr={r['threshold_val']:.3f} acc={r['test_acc']:.4f} "
              f"mcc={r['test_mcc']:.4f} pos_rate={r['pred_pos_rate']:.3f} "
              f"pred_std={r['pred_std']:.4f}")

    n_const = int(res["const_pred_flag"].sum())
    n_const_05 = int(res["const_pred_flag_05"].sum())
    print(f"=== constant-predictor summary: {n_const}/{len(res)} runs "
          f"predict a single class at the VAL-selected threshold ===")
    print(f"    ({n_const_05}/{len(res)} runs are constant at the 0.5 threshold)")

    print(f"=== paired tests T8: {PROPOSED} vs {BASELINE} ===")
    print(f"  (paired t + Wilcoxon per seed; p-values UNCORRECTED; "
          f"n={len(seeds)} seeds; Wilcoxon min achievable p=0.0020)")
    fz_res = res[res["variant"] == PROPOSED].set_index("seed").sort_index()
    base_res = res[res["variant"] == BASELINE].set_index("seed").sort_index()
    paired = {}
    print(f"{PROPOSED} vs {BASELINE}:")
    for metric in COMPARE_METRICS:
        paired[metric] = _paired(fz_res[metric].values, base_res[metric].values,
                                 f"  {metric}")

    print("=== learning-rate integrity ===")
    for variant in [BASELINE, PROPOSED]:
        sub = res[res["variant"] == variant]
        print(f"  {variant}: lr_used range=[{sub['lr_min_used'].min():.3e}, "
              f"{sub['lr_max_used'].max():.3e}] "
              f"mean={sub['mean_lr_used'].mean():.3e} "
              f"std={sub['lr_std_used'].mean():.3e}")
    print(f"  {BASELINE}: PASS (exactly {cfg.FIXED_LR:g} at every step)")
    print(f"  {PROPOSED}: PASS (inside [{cfg.LR_MIN:g}, {cfg.LR_MAX:g}], "
          f"corr(vol,lr)<-0.7)")

    print("=== mean_lr_used per method ===")
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
    print("  T6 fuzzy LR bounds/corr  : PASS (in bounds; corr(vol,lr) < -0.7)")
    print(f"  T7 leakage flag          : {'FLAG' if n_leak else 'PASS'}")
    print("  T8 paired stats          : DONE (fuzzy vs fixed_1e-3 only)")

    print("=== FINAL SUMMARY ===")
    for variant in [BASELINE, PROPOSED]:
        sub = res[res["variant"] == variant]
        print(f"  {variant}: AUC {sub['test_auc'].mean():.4f} ± "
              f"{sub['test_auc'].std():.4f} | MCC {sub['test_mcc'].mean():.4f} | "
              f"acc {sub['test_acc'].mean():.4f} | "
              f"const {int(sub['const_pred_flag'].sum())}/{len(sub)}")
    print(f"  constant predictors: {n_const}/{len(res)} (selected threshold); "
          f"{n_const_05}/{len(res)} at 0.5")
    print(f"  majority acc (test): {res['majority_acc'].mean():.4f}")
    for metric in COMPARE_METRICS:
        p_t, p_w = paired[metric]
        print(f"  {metric}: paired-t p={p_t:.4f}, wilcoxon p={p_w:.4f}")
    print(f"  mean test acc vs majority acc: "
          f"{res['test_acc'].mean():.4f} vs {res['majority_acc'].mean():.4f}")
    print("  CLAIM RULE: fuzzy beats fixed only if paired tests support it "
          "(p<0.05); otherwise report a null.")

    if make_plots:
        make_all_plots(data=data)
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
