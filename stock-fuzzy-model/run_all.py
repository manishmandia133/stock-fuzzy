"""Multi-seed experiment driver.

30 seeds x 5 LR variants -> results.csv + paired stats + plots + docs.

Primary metric (fixed in advance): test AUC.
The best fixed-LR baseline is selected on VALIDATION only. The paired
fuzzy_dyn vs best-fixed comparison uses bootstrap CI + paired-t + Wilcoxon;
Holm correction is applied across secondary metrics only. The claim is
decided by the primary metric alone.
"""
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

import config as cfg
import data as D
import fuzzy as fz
from evaluate import (
    COMPARE_METRICS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    epochs_to_target_val_loss,
    evaluate_run,
    holm_correction,
    paired_tests,
)
from plots import make_all_plots
from train import FIXED_VARIANTS, VARIANTS, train_variant

PROPOSED = "fuzzy_dyn"

# Old report expectations are tied to the ^GSPC/direction experiment only.
EXPECTED_GSPC_DIRECTION = {"n_train": 1900, "n_val": 391, "n_test": 392,
                           "up_train": 0.5400, "up_val": 0.5269,
                           "up_test": 0.5791,
                           "checksum": -3942.92227483448}


def _universe():
    return cfg.POOLED_TICKERS if cfg.UNIVERSE_MODE == "pooled" else cfg.UNIVERSE


def _fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _mean_std(sub, col, nd=4):
    return f"{sub[col].mean():.{nd}f} ± {sub[col].std():.{nd}f}"


def _write_docs(res, best_fixed, val_loss_target, stats, holm, claim_text,
                g, data, kept, t1_pass):
    """Write NOTES.md and REPORT_UPDATES.md from actual results only."""
    variants = list(res["variant"].unique())
    seeds = sorted(res["seed"].unique())
    lines = []
    lines.append("# NOTES.md — protocol, gate log, and actual run numbers\n")
    lines.append(f"Generated from `results.csv` on {datetime.now(UTC).isoformat()}.\n")
    lines.append("## Protocol (code facts)\n")
    lines.append(f"- Universe mode: `{cfg.UNIVERSE_MODE}`; contributing tickers: "
                 f"{len(kept)} (requested {len(_universe())}).")
    lines.append(f"- Target mode: `{cfg.TARGET_MODE}` "
                 f"({cfg.VOL_TARGET_SOURCE} vs TRAIN median for vol).")
    lines.append("- Chronological split; StandardScaler fit on TRAIN only; "
                 "clip to [-3, 3].")
    lines.append(f"- Window {cfg.WINDOW}; batch size {cfg.BATCH_SIZE}; max "
                 f"epochs {cfg.EPOCHS}; patience {cfg.PATIENCE}.")
    lines.append(f"- Early stopping: {cfg.EARLY_STOP_MA}-epoch moving-average "
                 f"validation loss; restore best weights with "
                 f"best_epoch >= {cfg.MIN_EPOCHS}.")
    lines.append(f"- Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds).")
    lines.append(f"- Variants: {', '.join(variants)}.")
    lines.append(f"- Best fixed LR selected on VALIDATION AUC only: "
                 f"`{best_fixed}`.")
    lines.append(f"- Secondary `epochs_to_target_val_loss` target = mean "
                 f"best-epoch validation loss of `{best_fixed}` = "
                 f"{_fmt(val_loss_target, 6)} (validation only).")
    lines.append("- T4 (fuzzy_vol curve shape) and T6 (LR bounds / "
                 "corr(vol,lr)) are tautological: they verify the controller "
                 "and schedule application, not training benefit.\n")

    lines.append("## Data / integrity\n")
    lines.append(f"- T1 data reproducibility: {'PASS' if t1_pass else 'FAIL'}.")
    lines.append("- T2 split/label spot-checks: PASS (asserts in data.py).")
    lines.append(f"- Split sizes: train {len(data['X_train'])} / "
                 f"val {len(data['X_val'])} / test {len(data['X_test'])}.")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        lines.append(f"- {name} positive rate: {y.mean():.4f} "
                     f"(majority {max(y.mean(), 1 - y.mean()):.4f}).")
    lines.append(f"- T3 windows/NaN: PASS (n_train="
                 f"{len(data['X_train'])}).")
    lines.append(f"- T4 fuzzy_vol curve: {'PASS' if g['pass'] else 'FAIL'} "
                 f"(surface mean {g['mean']:.4e}, max step "
                 f"{g['max_step']:.4e}; tautological).")
    lines.append("- T5 regime batching: all runs batch_vol_std > 0.15.")
    lines.append("- T6 fuzzy LR bounds: PASS; fuzzy_vol corr(vol,lr) < -0.7 "
                 "(tautological).")
    n_leak = int(res["leakage_flag"].sum())
    lines.append(f"- T7 leakage flag: {'FLAG' if n_leak else 'PASS'} "
                 f"({n_leak} run(s) with test AUC > 0.60).")
    lines.append("- T8 paired stats: reported below.\n")

    lines.append("## Results (mean ± std over seeds)\n")
    lines.append("| variant | test AUC (primary) | test log-loss | "
                 "epochs_to_target | AULC val loss | high-vol test AUC | "
                 "mean LR |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in variants:
        sub = res[res["variant"] == v]
        lines.append(
            f"| {v} | {_mean_std(sub, 'test_auc')} | "
            f"{_mean_std(sub, 'test_logloss')} | "
            f"{_mean_std(sub, 'epochs_to_target_val_loss', 2)} | "
            f"{_mean_std(sub, 'aulc_val_loss', 3)} | "
            f"{_mean_std(sub, 'test_auc_highvol')} | "
            f"{sub['mean_lr_used'].mean():.3e} |")
    lines.append("")
    lines.append("Across-seed std of primary test AUC: " + ", ".join(
        f"`{v}` {res.loc[res['variant'] == v, 'test_auc'].std():.4f}"
        for v in variants) + ".\n")

    lines.append(f"## Paired comparison: `{PROPOSED}` vs `{best_fixed}`\n")
    lines.append("| metric | mean diff | 95% bootstrap CI | paired-t p | "
                 "Wilcoxon p | Holm p (secondary) |")
    lines.append("|---|---|---|---|---|---|")
    for metric in COMPARE_METRICS:
        s = stats[metric]
        holm_p = "—" if metric == PRIMARY_METRIC else _fmt(holm["p_t"].get(metric), 4)
        lines.append(
            f"| {metric} | {_fmt(s['mean_diff'], 5)} | "
            f"[{_fmt(s['ci_lo'], 5)}, {_fmt(s['ci_hi'], 5)}] | "
            f"{_fmt(s['p_t'], 4)} | {_fmt(s['p_w'], 4)} | {holm_p} |")
    lines.append("")
    lines.append(f"**Claim rule:** primary paired-t p < 0.05 AND 95% CI "
                 f"excludes 0 AND direction favours `{PROPOSED}`; otherwise "
                 f"NULL RESULT. (Wilcoxon is reported alongside but the "
                 f"pre-specified primary test is paired-t.)")
    lines.append(f"\n**Outcome:** {claim_text}\n")

    with open("NOTES.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    rep = []
    rep.append("# REPORT_UPDATES.md — required report changes "
               "(actual run numbers only)\n")
    rep.append(f"Source: `python prepare_data.py && python main.py`, "
               f"{datetime.now(UTC).date()}. Values come from `results.csv` "
               f"and the generated logs.\n")
    rep.append("## Data and protocol\n")
    rep.append(f"- Universe: `{cfg.UNIVERSE_MODE}` ({len(kept)} usable "
               f"tickers out of {len(_universe())} requested).")
    rep.append(f"- Target: `{cfg.TARGET_MODE}`; vol label uses next-day "
               f"{cfg.VOL_TARGET_SOURCE} vs TRAIN median.")
    rep.append("- Chronological split with one set of cut dates for all "
               "tickers in pooled mode; scaler/median/vol_rank from train only.")
    rep.append(f"- Windows: train/val/test "
               f"{len(data['X_train'])}/{len(data['X_val'])}/"
               f"{len(data['X_test'])}.")
    rep.append(f"- Early stopping on {cfg.EARLY_STOP_MA}-epoch moving-average "
               f"validation loss; best_epoch >= {cfg.MIN_EPOCHS}; best weights "
               f"restored.")
    rep.append(f"- Fixed LRs: {', '.join(FIXED_VARIANTS)}. Best fixed by "
               f"validation AUC: `{best_fixed}`.")
    rep.append("- Fuzzy variants: `fuzzy_dyn` (per-epoch val-loss change + "
               "vol_rank, 3-epoch warmup) and `fuzzy_vol` (per-batch "
               "vol_rank curve, ablation).\n")
    rep.append("## Results\n")
    rep.append("| variant | test AUC | test log-loss | epochs_to_target | "
               "AULC | high-vol AUC |")
    rep.append("|---|---|---|---|---|---|")
    for v in variants:
        sub = res[res["variant"] == v]
        rep.append(
            f"| {v} | {_mean_std(sub, 'test_auc')} | "
            f"{_mean_std(sub, 'test_logloss')} | "
            f"{_mean_std(sub, 'epochs_to_target_val_loss', 2)} | "
            f"{_mean_std(sub, 'aulc_val_loss', 3)} | "
            f"{_mean_std(sub, 'test_auc_highvol')} |")
    rep.append("")
    rep.append("## Paired statistics\n")
    for metric in COMPARE_METRICS:
        s = stats[metric]
        tag = "primary, uncorrected" if metric == PRIMARY_METRIC else \
            f"secondary, Holm p={_fmt(holm['p_t'].get(metric), 4)}"
        rep.append(f"- `{metric}` ({tag}): mean diff {_fmt(s['mean_diff'], 5)}, "
                   f"95% CI [{_fmt(s['ci_lo'], 5)}, {_fmt(s['ci_hi'], 5)}], "
                   f"paired-t p={_fmt(s['p_t'], 4)}, "
                   f"Wilcoxon p={_fmt(s['p_w'], 4)}.")
    rep.append("")
    rep.append(f"## Claim\n\n{claim_text}\n")
    rep.append("T1..T8: T1 " + ("PASS" if t1_pass else "FAIL") +
               ", T2 PASS, T3 PASS, T4 " + ("PASS" if g["pass"] else "FAIL") +
               " (tautological), T5 PASS, T6 PASS (tautological), T7 " +
               ("FLAG" if n_leak else "PASS") + ", T8 DONE.\n")
    with open("REPORT_UPDATES.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))


def main(seeds=None, variants=None, make_plots=True):
    seeds = list(seeds or cfg.SEEDS)
    variants = list(variants or VARIANTS)

    print("=== fuzzy gate (T4, Mamdani fuzzy_vol curve) ===")
    g = fz.check_gate()
    for k, v in g.items():
        print(f"  {k}: {v}")
    print("  NOTE: T4/T6 are tautological — they verify the controller shape "
          "and schedule application, not training benefit.")
    if not g["pass"]:
        print("T4 GATE FAILED — fix fuzzy.py before training.")
        sys.exit(1)
    print("  T4: PASS")

    print("=== data ===")
    tickers = _universe()
    print(f"universe_mode={cfg.UNIVERSE_MODE} target_mode={cfg.TARGET_MODE} "
          f"tickers_requested={len(tickers)}")
    raw = D.batch_download(tickers, cfg.START_DATE, cfg.END_DATE)
    engineered = {t: D.engineer_features(df) for t, df in raw.items()}
    engineered = {t: df for t, df in engineered.items()
                  if len(df) > cfg.WINDOW + 20}
    print(f"tickers with engineered features: {len(engineered)}")
    data, kept = D.build_dataset(engineered)
    print(f"kept tickers: {len(kept)}")
    print("T2: PASS (chronological split asserts + label spot-checks)")
    print(f"T3: PASS (n_train={len(data['X_train'])} > 0, no NaN)")

    print("=== split stats ===")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        print(f"  {name}: n={len(y)} pos_rate={y.mean():.4f} "
              f"majority={max(y.mean(), 1 - y.mean()):.4f}")

    print("=== reproducibility check (T1: data build twice) ===")
    h1 = float(data["X_train"].sum())
    data2, _ = D.build_dataset(engineered)
    h2 = float(data2["X_train"].sum())
    t1_pass = (h1 == h2)
    print(f"  T1: {'PASS' if t1_pass else 'FAIL'} (checksum={h1:.4f})")
    assert t1_pass, "non-deterministic data pipeline"
    del data2

    if cfg.UNIVERSE_MODE == "gspc" and cfg.TARGET_MODE == "direction":
        print("=== report expected-vs-actual (Sec 6.1, GSPC/direction) ===")
        actual = {"n_train": len(data["X_train"]),
                  "n_val": len(data["X_val"]),
                  "n_test": len(data["X_test"]),
                  "up_train": float(data["y_train"].mean()),
                  "up_val": float(data["y_val"].mean()),
                  "up_test": float(data["y_test"].mean()),
                  "checksum": h1}
        for k, e in EXPECTED_GSPC_DIRECTION.items():
            a = actual[k]
            match = (a == e) if isinstance(e, int) else abs(a - e) < 0.0005
            flag = "MATCHES REPORT" if match else \
                f"DIFFERS FROM REPORT (actual {a} vs report {e})"
            print(f"  {k}: actual={a} expected={e} -> {flag}")
    else:
        print("=== report expected-vs-actual: not applicable "
              f"({cfg.UNIVERSE_MODE}/{cfg.TARGET_MODE}) ===")

    rows = []
    histories = {}
    for seed in seeds:
        for variant in variants:
            model, history, log_df, info = train_variant(
                data["X_train"], data["y_train"], data["vr_train"],
                data["X_val"], data["y_val"], variant, seed)

            if info["batch_vol_std"] <= 0.15:
                print(f"T5 GATE FAILED for {variant} s{seed}: "
                      f"batch vol std={info['batch_vol_std']:.3f} <= 0.15")
                sys.exit(1)
            if variant == "fuzzy_vol" and not (info["vol_lr_corr"] < -0.7):
                print(f"T6 GATE FAILED: corr(vol,lr)={info['vol_lr_corr']:.3f} "
                      f">= -0.7.")
                sys.exit(1)

            m = evaluate_run(
                model, history, log_df,
                data["X_val"], data["y_val"],
                data["X_test"], data["y_test"],
                data["X_train"][:, -1, :], data["y_train"],
                seed=seed, vr_test=data["vr_test"],
                best_epoch=info["best_epoch"])
            if m.get("leakage_flag"):
                print(f"WARNING: AUC={m['test_auc']:.3f} > 0.60 suspect leakage "
                      f"({variant} s{seed}) — flagged, not dropped.")
            if m.get("const_pred_flag"):
                print(f"NOTE: {variant} s{seed} is a constant predictor at "
                      f"thr={m['threshold_val']:.3f} "
                      f"(pos_rate={m['pred_pos_rate']:.3f}).")
            rows.append({"seed": seed, "variant": variant, **info, **m})
            histories[(variant, seed)] = history

    res = pd.DataFrame(rows)

    # ---- experimental-integrity checks -----------------------------------
    assert set(res["variant"]) == set(variants), \
        f"unexpected variants in results: {sorted(set(res['variant']))}"
    for seed in seeds:
        seed_rows = res[res["seed"] == seed]
        assert set(seed_rows["variant"]) == set(variants), \
            f"seed {seed} is missing a variant"
        assert seed_rows["n_train"].nunique() == 1, \
            f"seed {seed}: train size differs across methods"
        assert seed_rows["n_val"].nunique() == 1, \
            f"seed {seed}: val size differs across methods"

    for name, lr in cfg.FIXED_LRS.items():
        sub = res[res["variant"] == name]
        if len(sub) == 0:
            continue
        assert bool((sub["lr_min_used"] == lr).all()), \
            f"{name} did not use exactly {lr}"
        assert bool((sub["lr_max_used"] == lr).all()), \
            f"{name} changed its learning rate"
        assert bool((sub["lr_std_used"] == 0.0).all()), \
            f"{name} learning rate has non-zero variance"
    for name in cfg.FUZZY_VARIANTS:
        sub = res[res["variant"] == name]
        if len(sub) == 0:
            continue
        assert bool((sub["lr_min_used"] >= cfg.LR_MIN).all()), \
            f"{name} LR below bounds"
        assert bool((sub["lr_max_used"] <= cfg.LR_MAX).all()), \
            f"{name} LR above bounds"

    # ---- best fixed LR: VALIDATION ONLY ----------------------------------
    fixed_rows = res[res["variant"].isin(FIXED_VARIANTS)]
    val_rank = (fixed_rows.groupby("variant")["best_val_auc"].mean()
                .sort_values(ascending=False))
    print("=== fixed-LR selection (mean validation AUC, no test data) ===")
    for name, val in val_rank.items():
        print(f"  {name}: val AUC={val:.4f}")
    best_fixed = str(val_rank.index[0])
    print(f"  selected best fixed: {best_fixed}")

    # Target for epochs_to_target_val_loss uses validation only.
    val_loss_target = float(
        res.loc[res["variant"] == best_fixed, "best_val_loss"].mean())
    res["epochs_to_target_val_loss"] = [
        epochs_to_target_val_loss(histories[(r["variant"], int(r["seed"]))],
                                  val_loss_target)
        for _, r in res.iterrows()
    ]
    print(f"  val-loss target for epochs_to_target: {val_loss_target:.6f} "
          f"(mean best-epoch val loss of {best_fixed})")

    res.to_csv("results.csv", index=False)

    # ---- paired stats -----------------------------------------------------
    fz_res = res[res["variant"] == PROPOSED].set_index("seed").sort_index()
    bf_res = res[res["variant"] == best_fixed].set_index("seed").sort_index()
    stats = {}
    for metric in COMPARE_METRICS:
        stats[metric] = paired_tests(fz_res[metric].values,
                                     bf_res[metric].values)
    sec_p_t = [stats[m]["p_t"] for m in SECONDARY_METRICS]
    sec_p_w = [stats[m]["p_w"] for m in SECONDARY_METRICS]
    holm = {"p_t": dict(zip(SECONDARY_METRICS, holm_correction(sec_p_t))),
            "p_w": dict(zip(SECONDARY_METRICS, holm_correction(sec_p_w)))}

    primary = stats[PRIMARY_METRIC]
    ci_excludes_zero = (
        np.isfinite(primary.get("ci_lo", np.nan)) and
        np.isfinite(primary.get("ci_hi", np.nan)) and
        (primary["ci_lo"] > 0.0 or primary["ci_hi"] < 0.0))
    favors_fuzzy = primary["mean_diff"] > 0.0
    claim = (np.isfinite(primary["p_t"]) and primary["p_t"] < 0.05 and
             ci_excludes_zero and favors_fuzzy)
    if claim:
        claim_text = (f"CLAIM: `{PROPOSED}` beats `{best_fixed}` on the "
                      f"primary metric (test AUC): mean diff "
                      f"{primary['mean_diff']:+.4f}, 95% CI "
                      f"[{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}], "
                      f"paired-t p={primary['p_t']:.4f}.")
    else:
        claim_text = (f"NULL RESULT: `{PROPOSED}` vs `{best_fixed}` primary "
                      f"test-AUC mean diff {primary['mean_diff']:+.4f}, "
                      f"95% CI [{primary['ci_lo']:+.4f}, "
                      f"{primary['ci_hi']:+.4f}], paired-t "
                      f"p={primary['p_t']:.4f}. No fuzzy-beats-fixed claim.")

    with open("run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_utc": datetime.now(UTC).isoformat(),
            "methods": list(variants),
            "seeds": list(seeds),
            "best_fixed_validation_selected": best_fixed,
            "val_loss_target": val_loss_target,
            "config": {
                "universe_mode": cfg.UNIVERSE_MODE,
                "universe_requested": len(_universe()),
                "universe_used": len(kept),
                "start_date": cfg.START_DATE,
                "end_date": cfg.END_DATE, "window": cfg.WINDOW,
                "batch_size": cfg.BATCH_SIZE, "max_epochs": cfg.EPOCHS,
                "patience": cfg.PATIENCE, "min_epochs": cfg.MIN_EPOCHS,
                "early_stop_metric": f"val_loss_ma{cfg.EARLY_STOP_MA}",
                "warmup_epochs": cfg.WARMUP_EPOCHS,
                "grad_clip": cfg.GRAD_CLIP,
                "weight_decay": cfg.WEIGHT_DECAY,
                "hidden_size": cfg.HIDDEN_SIZE, "num_layers": cfg.NUM_LAYERS,
                "dropout": cfg.DROPOUT, "clip": cfg.CLIP,
                "threshold_grid": [cfg.THRESHOLD_LOW, cfg.THRESHOLD_HIGH,
                                   cfg.THRESHOLD_STEP],
                "train_frac": cfg.TRAIN_FRAC, "val_frac": cfg.VAL_FRAC,
                "features": cfg.FEATURE_COLS,
                "target_mode": cfg.TARGET_MODE,
                "target": cfg.TARGET_COL,
                "vol_target_source": cfg.VOL_TARGET_SOURCE,
                "fixed_lrs": cfg.FIXED_LRS,
                "fuzzy_lr_min": cfg.LR_MIN,
                "fuzzy_lr_max": cfg.LR_MAX,
                "fuzzy_centroids": [cfg.LR_SLOW_C, cfg.LR_MED_C,
                                    cfg.LR_FAST_C],
                "fuzzy_dyn_dloss_scale": cfg.FUZZY_DYN_DLOSS_SCALE,
            },
        }, f, indent=2)

    # ---- reporting --------------------------------------------------------
    print("=== results: mean ± std per variant ===")
    show_cols = COMPARE_METRICS + ["test_acc", "test_mcc", "mean_lr_used",
                                   "best_epoch", "epochs_run"]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(res.groupby("variant")[show_cols].agg(["mean", "std"]).to_string())

    print("=== test AUC per seed + bootstrap 95% CI ===")
    for _, r in res.iterrows():
        print(f"  {r['variant']} s{int(r['seed'])}: "
              f"AUC={r['test_auc']:.4f} [{r['test_auc_lo']:.4f}, "
              f"{r['test_auc_hi']:.4f}] logloss={r['test_logloss']:.4f} "
              f"highvolAUC={r['test_auc_highvol']:.4f} "
              f"lr={r['mean_lr_used']:.3e}")

    print(f"=== paired tests: {PROPOSED} vs {best_fixed} ===")
    print(f"  n={len(seeds)} seeds; primary uncorrected; secondary Holm-corrected")
    for metric in COMPARE_METRICS:
        s = stats[metric]
        tag = "primary" if metric == PRIMARY_METRIC else \
            f"Holm p={_fmt(holm['p_t'].get(metric), 4)}"
        print(f"  {metric}: diff={s['mean_diff']:+.5f} "
              f"CI=[{s['ci_lo']:+.5f}, {s['ci_hi']:+.5f}] "
              f"t p={s['p_t']:.4f} w p={s['p_w']:.4f} ({tag})")

    print("=== learning-rate integrity ===")
    for variant in variants:
        sub = res[res["variant"] == variant]
        print(f"  {variant}: lr_used range=[{sub['lr_min_used'].min():.3e}, "
              f"{sub['lr_max_used'].max():.3e}] "
              f"mean={sub['mean_lr_used'].mean():.3e} "
              f"std={sub['lr_std_used'].mean():.3e}")

    print("=== across-seed primary std ===")
    for variant in variants:
        sub = res[res["variant"] == variant]
        print(f"  {variant}: test AUC {sub['test_auc'].mean():.4f} ± "
              f"{sub['test_auc'].std():.4f}")

    n_leak = int(res["leakage_flag"].sum())
    print("=== GATE SUMMARY T1..T8 ===")
    print(f"  T1 data reproducibility : {'PASS' if t1_pass else 'FAIL'}")
    print("  T2 split/label checks    : PASS (asserts in data.py)")
    print(f"  T3 scope windows/NaN     : PASS (n_train={len(data['X_train'])})")
    print(f"  T4 fuzzy_vol curve       : {'PASS' if g['pass'] else 'FAIL'} "
          f"(tautological: shape only)")
    print("  T5 regime batching       : PASS (all batch_vol_std > 0.15)")
    print("  T6 fuzzy LR bounds       : PASS "
          "(tautological: bounds + fuzzy_vol corr)")
    print(f"  T7 leakage flag          : {'FLAG' if n_leak else 'PASS'}")
    print(f"  T8 paired stats          : DONE ({PROPOSED} vs {best_fixed})")

    print("=== FINAL SUMMARY ===")
    for variant in variants:
        sub = res[res["variant"] == variant]
        print(f"  {variant}: AUC {sub['test_auc'].mean():.4f} ± "
              f"{sub['test_auc'].std():.4f} | "
              f"logloss {sub['test_logloss'].mean():.4f} | "
              f"highvol AUC {sub['test_auc_highvol'].mean():.4f} | "
              f"mean LR {sub['mean_lr_used'].mean():.3e}")
    print(f"  best fixed (validation): {best_fixed}")
    print(f"  {claim_text}")
    print("  CLAIM RULE: only the primary metric (test AUC) can decide the "
          "claim; secondary metrics are Holm-corrected and descriptive.")

    _write_docs(res, best_fixed, val_loss_target, stats, holm, claim_text,
                g, data, kept, t1_pass)
    print("Wrote NOTES.md and REPORT_UPDATES.md from actual results.")

    if make_plots:
        make_all_plots(data=data)
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
