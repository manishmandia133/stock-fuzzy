"""Multi-seed experiment driver: 2 fixed-LR grid + 2 fuzzy variants.

BEST_FIXED is selected using VALIDATION loss only (lowest mean best_val_loss
across seeds). Test metrics are used only for the final paired comparison.
"""

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

import config as cfg
import data as D
import evaluate as ev
import fuzzy as fz
from evaluate import evaluate_run
from plots import make_all_plots
from train import FUZZY_VARIANTS, VARIANTS, train_variant

PROPOSED = "fuzzy"
ABLATION = "fuzzy_vol_only"
COMPARE_METRICS = [
    "test_auc",
    "test_acc",
    "test_mcc",
    "best_val_loss",
    "highvol_loss_var",
    "train_val_gap",
]

# Report expectations: compared against actuals at runtime.
EXPECTED = {
    "n_train": 1893,
    "n_val": 382,
    "n_test": 383,
    "up_train": 0.5404,
    "up_val": 0.5236,
    "up_test": 0.5770,
    "checksum": 102657.2367,
}


def _check_controller() -> dict[str, float | bool]:
    """Run the fuzzy shape/bounds gate and exit on failure."""
    print("=== fuzzy gate (T4, two-input Mamdani FIS) ===")
    gate = fz.check_gate()
    for key, value in gate.items():
        print(f"  {key}: {value}")
    print("  NOTE: controller-shape gates verify the schedule, not training benefit.")
    if not gate["pass"]:
        print("T4 GATE FAILED — fix fuzzy.py before training.")
        sys.exit(1)
    print("  T4: PASS")
    return gate


def _load_data() -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Download, engineer, and window the data; print T2/T3 status."""
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
    return data, engineered


def _reproducibility_check(
    data: dict[str, np.ndarray],
    engineered: dict[str, pd.DataFrame],
) -> tuple[float, bool]:
    """Rebuild the data and assert the train checksum is deterministic."""
    print("=== reproducibility check (T1: data build twice) ===")
    h1 = float(data["X_train"].sum())
    data2, _ = D.build_dataset(engineered)
    h2 = float(data2["X_train"].sum())
    t1_pass = h1 == h2
    print(
        f"  T1: {'PASS' if t1_pass else 'FAIL'} "
        f"(checksum={h1:.4f}, expected={EXPECTED['checksum']:.4f})"
    )
    assert t1_pass, "non-deterministic data pipeline"
    return h1, t1_pass


def _report_expected(data: dict[str, np.ndarray], checksum: float) -> None:
    """Compare actual split statistics with the fixed report values."""
    print("=== report expected-vs-actual ===")
    actual = {
        "n_train": len(data["X_train"]),
        "n_val": len(data["X_val"]),
        "n_test": len(data["X_test"]),
        "up_train": float(data["y_train"].mean()),
        "up_val": float(data["y_val"].mean()),
        "up_test": float(data["y_test"].mean()),
        "checksum": checksum,
    }
    for key in [
        "n_train",
        "n_val",
        "n_test",
        "up_train",
        "up_val",
        "up_test",
        "checksum",
    ]:
        expected, got = EXPECTED[key], actual[key]
        match = (
            (got == expected)
            if isinstance(expected, int)
            else abs(got - expected) < 0.0005
        )
        flag = (
            "MATCHES REPORT"
            if match
            else f"DIFFERS FROM REPORT (actual {got} vs report {expected})"
        )
        print(f"  {key}: actual={got} expected={expected} -> {flag}")


def _train_all(
    data: dict[str, np.ndarray],
    seeds: Sequence[int],
    variants: Sequence[str],
) -> pd.DataFrame:
    """Train every seed/variant, evaluate it, and return the results frame."""
    rows = []
    for seed in seeds:
        for variant in variants:
            model, history, log_df, info = train_variant(
                data["X_train"],
                data["y_train"],
                data["vr_train"],
                data["X_val"],
                data["y_val"],
                variant,
                seed,
            )

            if info["batch_vol_std"] <= 0.15:
                print(
                    f"T5 GATE FAILED for {variant} s{seed}: "
                    f"batch vol std={info['batch_vol_std']:.3f} <= 0.15"
                )
                sys.exit(1)
            if variant == ABLATION and not (info["vol_lr_corr"] < -0.7):
                print(
                    f"T6 GATE FAILED: {variant} corr(vol,lr)="
                    f"{info['vol_lr_corr']:.3f} >= -0.7."
                )
                sys.exit(1)

            metrics = evaluate_run(
                model,
                history,
                log_df,
                data["X_test"],
                data["y_test"],
                data["X_train"][:, -1, :],
                data["y_train"],
                seed=seed,
            )
            if metrics.get("leakage_flag"):
                print(
                    f"WARNING: AUC={metrics['test_auc']:.3f} > 0.60 suspect "
                    f"leakage ({variant} s{seed}) — flagged, not dropped."
                )
            if metrics.get("const_pred_flag"):
                print(
                    f"NOTE: {variant} s{seed} is a constant predictor "
                    f"(pos_rate={metrics['pred_pos_rate']:.3f}, "
                    f"pred_std={metrics['pred_std']:.4f})."
                )
            rows.append({"seed": seed, "variant": variant, **info, **metrics})
            if (
                isinstance(metrics.get("test_auc"), float)
                and abs(history["val_loss"][-1] - np.log(2)) < 0.01
            ):
                print(
                    f"NOTE: {variant} s{seed} val_loss ~ ln2 "
                    f"({history['val_loss'][-1]:.4f}); model may not be learning."
                )
    return pd.DataFrame(rows)


def _check_integrity(res: pd.DataFrame, seeds: Sequence[int]) -> None:
    """Assert every seed has all variants and every LR contract holds."""
    assert set(res["variant"]) == set(
        VARIANTS
    ), f"unexpected variants in results: {sorted(set(res['variant']))}"
    for seed in seeds:
        seed_rows = res[res["seed"] == seed]
        assert set(seed_rows["variant"]) == set(
            VARIANTS
        ), f"seed {seed} does not contain all variants"
        assert (
            seed_rows["n_train"].nunique() == 1
        ), f"seed {seed}: train size differs across variants"
        assert (
            seed_rows["n_val"].nunique() == 1
        ), f"seed {seed}: val size differs across variants"

    # Fixed variants must use exactly their configured LR.
    for name, expected_lr in cfg.FIXED_LR_VARIANTS.items():
        sub = res[res["variant"] == name]
        assert bool(
            (sub["lr_min_used"] == expected_lr).all()
        ), f"{name} did not use exactly {expected_lr:g}"
        assert bool(
            (sub["lr_max_used"] == expected_lr).all()
        ), f"{name} changed its learning rate"
        assert bool(
            (sub["lr_std_used"] == 0.0).all()
        ), f"{name} learning rate has non-zero variance"

    # Fuzzy variants must vary and stay in bounds.
    for name in FUZZY_VARIANTS:
        sub = res[res["variant"] == name]
        assert bool((sub["lr_min_used"] >= cfg.LR_MIN).all()), f"{name} LR below bounds"
        assert bool((sub["lr_max_used"] <= cfg.LR_MAX).all()), f"{name} LR above bounds"
        assert bool((sub["lr_std_used"] > 0.0).all()), f"{name} LR never changed"

    # The vol-only ablation must remain strongly inverse to volatility. The
    # 2-input controller also moves LR with the trend, so its *global* corr is
    # reported rather than used as a gate; its monotonicity is checked by
    # fuzzy.check_gate() on the surface.
    ablation = res[res["variant"] == ABLATION]
    assert bool(
        (ablation["vol_lr_corr"] < -0.7).all()
    ), "fuzzy_vol_only LR did not respond inversely to volatility (T6)"


def _print_best_fixed(res: pd.DataFrame) -> str:
    """Print the validation-only fixed-LR grid selection."""
    best_fixed, means = ev.select_best_fixed(res)
    print("=== fixed-LR grid selection (validation only) ===")
    for name, value in means.items():
        print(f"  {name}: mean best_val_loss={value:.6f}")
    print(f"  BEST_FIXED = {best_fixed} (lowest mean validation loss)")
    return best_fixed


def _write_outputs(
    res: pd.DataFrame,
    seeds: Sequence[int],
    best_fixed: str,
) -> None:
    """Write the results table and reproducibility metadata."""
    res.to_csv("results.csv", index=False)
    with open("run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "variants": VARIANTS,
                "seeds": list(seeds),
                "best_fixed_rule": (
                    "lowest mean best_val_loss across seeds; "
                    "validation only, never test"
                ),
                "best_fixed": best_fixed,
                "config": {
                    "universe": cfg.UNIVERSE,
                    "start_date": cfg.START_DATE,
                    "end_date": cfg.END_DATE,
                    "window": cfg.WINDOW,
                    "hidden": cfg.HIDDEN,
                    "num_layers": cfg.NUM_LAYERS,
                    "dropout": cfg.DROPOUT,
                    "batch_size": cfg.BATCH_SIZE,
                    "max_epochs": cfg.EPOCHS,
                    "patience": cfg.PATIENCE,
                    "grad_clip": cfg.GRAD_CLIP,
                    "train_frac": cfg.TRAIN_FRAC,
                    "val_frac": cfg.VAL_FRAC,
                    "features": cfg.FEATURE_COLS,
                    "target": cfg.TARGET_COL,
                    "fixed_lr_grid": cfg.FIXED_LR_GRID,
                    "fixed_lr_variants": cfg.FIXED_LR_VARIANTS,
                    "fuzzy_lr_min": cfg.LR_MIN,
                    "fuzzy_lr_max": cfg.LR_MAX,
                    "trend_scale": cfg.TREND_SCALE,
                    "trend_clip": [cfg.TREND_MIN, cfg.TREND_MAX],
                },
            },
            handle,
            indent=2,
        )


def _print_variant_summary(res: pd.DataFrame) -> None:
    """Print the mean ± std table over seeds."""
    print("=== per-variant results (mean ± std over seeds) ===")
    summary = ev.summary_table(res, VARIANTS)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_auc(res: pd.DataFrame) -> None:
    """Print per-run AUC with bootstrap CIs."""
    print("=== test AUC per seed + bootstrap 95% CI ===")
    for _, row in res.iterrows():
        print(
            f"  {row['variant']} s{int(row['seed'])}: "
            f"AUC={row['test_auc']:.4f} "
            f"[{row['test_auc_lo']:.4f}, {row['test_auc_hi']:.4f}] "
            f"pos_rate={row['pred_pos_rate']:.3f} "
            f"pred_std={row['pred_std']:.4f}"
        )


def _print_const_summary(res: pd.DataFrame) -> None:
    """Print how many runs collapse to a single predicted class."""
    n_const = int(res["const_pred_flag"].sum())
    print(
        f"=== constant-predictor summary: {n_const}/{len(res)} runs "
        f"predict a single class at 0.5 threshold ==="
    )


def _print_paired(res: pd.DataFrame, best_fixed: str) -> None:
    """Print paired t/Wilcoxon tests for fuzzy variants vs BEST_FIXED."""
    print(
        f"=== paired tests T8: {PROPOSED} / {ABLATION} vs "
        f"BEST_FIXED ({best_fixed}) ==="
    )
    print(
        "  (paired t + Wilcoxon per seed; p-values UNCORRECTED; "
        "n=5 seeds; Wilcoxon min achievable p=0.0625)"
    )
    base = res[res["variant"] == best_fixed].set_index("seed").sort_index()
    for fuzzy_name in [PROPOSED, ABLATION]:
        fuzzy = res[res["variant"] == fuzzy_name].set_index("seed").sort_index()
        print(f"{fuzzy_name} vs {best_fixed}:")
        for metric in COMPARE_METRICS:
            p_t, p_w, n_pairs = ev.paired_test(
                fuzzy[metric].values, base[metric].values
            )
            if n_pairs < 2:
                print(f"  {metric}: not enough paired observations")
            else:
                print(f"  {metric}: paired-t p={p_t:.4f} | " f"wilcoxon p={p_w:.4f}")


def _print_lr_integrity(res: pd.DataFrame) -> None:
    """Print applied-LR ranges and trend ranges per variant."""
    print("=== learning-rate integrity ===")
    for variant in VARIANTS:
        sub = res[res["variant"] == variant]
        print(
            f"  {variant}: lr_used range=[{sub['lr_min_used'].min():.3e}, "
            f"{sub['lr_max_used'].max():.3e}] "
            f"mean={sub['mean_lr_used'].mean():.3e} "
            f"std={sub['lr_std_used'].mean():.3e} "
            f"trend=[{sub['trend_min_used'].min():.3f}, "
            f"{sub['trend_max_used'].max():.3f}]"
        )
    print("  fixed variants: PASS (exact configured LR at every step)")
    print(
        f"  fuzzy variants: PASS (inside [{cfg.LR_MIN:g}, {cfg.LR_MAX:g}]; "
        f"vol-only corr(vol,lr)<-0.7; 2-input surface monotone in vol)"
    )


def _print_mean_lr(res: pd.DataFrame) -> None:
    """Print per-method mean LR summary statistics."""
    print("=== mean_lr_used per method ===")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(
            res.groupby("variant")[["mean_lr_used"]]
            .agg(["mean", "std", "min", "max"])
            .to_string()
        )


def _print_gate_summary(
    t1_pass: bool,
    data: dict[str, np.ndarray],
    gate: dict[str, float | bool],
    n_leak: int,
    best_fixed: str,
) -> None:
    """Print the final T1..T8 gate table."""
    print("=== GATE SUMMARY T1..T8 ===")
    print(f"  T1 data reproducibility : {'PASS' if t1_pass else 'FAIL'}")
    print("  T2 split/label checks    : PASS (asserts in data.py)")
    print(f"  T3 scope windows/NaN     : PASS (n_train={len(data['X_train'])})")
    print(
        f"  T4 fuzzy surface         : "
        f"{'PASS' if gate['pass'] else 'FAIL'} (tautological: shape only)"
    )
    print("  T5 regime batching       : PASS (all batch_vol_std > 0.15)")
    print(
        "  T6 fuzzy LR bounds/corr  : PASS (bounds; vol-only corr<-0.7; "
        "fuzzy surface monotone)"
    )
    print(f"  T7 leakage flag          : {'FLAG' if n_leak else 'PASS'}")
    print(f"  T8 paired stats          : DONE (fuzzy vs {best_fixed})")


def main(
    seeds: Sequence[int] | None = None,
    variants: Sequence[str] | None = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Run the full experiment and return the results frame."""
    seeds = list(seeds or cfg.SEEDS)
    variants = list(variants or VARIANTS)

    gate = _check_controller()
    data, engineered = _load_data()

    print("=== split stats ===")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        print(
            f"  {name}: n={len(y)} up_rate={y.mean():.4f} "
            f"majority={max(y.mean(), 1 - y.mean()):.4f}"
        )

    checksum, t1_pass = _reproducibility_check(data, engineered)
    _report_expected(data, checksum)

    res = _train_all(data, seeds, variants)
    _check_integrity(res, seeds)
    best_fixed = _print_best_fixed(res)
    _write_outputs(res, seeds, best_fixed)

    _print_variant_summary(res)
    _print_auc(res)
    _print_const_summary(res)
    _print_paired(res, best_fixed)
    _print_lr_integrity(res)
    _print_mean_lr(res)

    n_leak = int(res["leakage_flag"].sum())
    if n_leak:
        print(
            f"T7: FLAG — {n_leak} run(s) with AUC>0.60 "
            f"(suspect leakage, flagged not dropped)"
        )
    else:
        print("T7: PASS (no test AUC > 0.60)")
    print(
        "T8: DONE (paired stats above; do not claim fuzzy wins unless tests "
        "support it)"
    )

    _print_gate_summary(t1_pass, data, gate, n_leak, best_fixed)

    if make_plots:
        make_all_plots(data=data)
    print("Done. results.csv written.")
    return res


if __name__ == "__main__":
    main()
