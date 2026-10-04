# NOTES.md — gate log (PLAN: document all gate passes/failures here)

## T1 Reproducibility
- `config.set_seed(s)` seeds random/numpy/torch/cuda. No `perm` pre-shuffle;
  train windows are sorted by `end_date`, batch ORDER permuted per epoch.
- Gate: data build twice -> identical sums. (see run_all.py T1 check)

## T2 Features
- Stationary: return_1d, hl_range, oc_ret, vol_chg, volatility_14d.
- Assertions: chronological split per ticker; 3-window label spot-check;
  class balance + majority baseline printed.

## T3 Scope
- `UNIVERSE = ["^GSPC"]`, 70/15/15 split, `END_DATE = "2025-12-31"` fixed.
- Gate: n_train > 1500 windows; no NaN.

## T4 Fuzzy controller
- Lookup table (201-pt grid), no per-batch skfuzzy, no try/except.
- Gate: smooth (max step < 5% range), mean within 20% of 1e-3, monotonic decreasing.

## T5 Regime batching
- Gate: std of batch vol across epoch > 0.15. Exits non-zero on failure.

## T6 Training
- 7 variants, per-batch logs to `logs/{variant}_s{seed}.csv`,
  best-val-loss weights, grad clip 1.0, patience 5.
- Gate (fuzzy): corr(vol, lr) < -0.7 else STOP.

## T7 Evaluation
- AUC > 0.60 -> `leakage_flag`, flagged not dropped. LogReg-on-last-day ~0.50 expected.

## T8 Stats
- Seeds 0-4 x variants -> results.csv; paired t + Wilcoxon fuzzy vs
  fixed_1e-3 and vs fixed_mean_fuzzy. p-values reported as-is.

## Run log — ^GSPC 2015-01-01..2025-12-31, seeds 0-4 x 7 variants (2026-10-04)
- T1 PASS (deterministic data build), T2 PASS (assertions), T3 PASS
  (n_train=1863 > 1500, no NaN), T4 PASS
  (max_step=1.8e-05, range=1.1e-03, mean=9.63e-04, monotonic),
  T5 PASS (batch vol std=0.176 > 0.15), T6 PASS (fuzzy corr(vol,lr)≈-0.99),
  T7 PASS (no AUC > 0.60; logreg_auc=0.509).
- Result: test AUC ≈ 0.532 ± 0.024 all variants; fuzzy vs fixed_1e-3
  paired-t p=0.40, vs fixed_mean_fuzzy p=0.30 → honest null on AUC.
- All models predict the majority class (test_acc == majority 0.578, MCC=0.0);
  signal is ranking-only (AUC>0.5) with no usable 0.5-threshold edge.
- Stability: highvol_loss_var paired-t p=0.0047 (fuzzy vs fixed_1e-3),
  p=0.0073 (vs fixed_mean_fuzzy); Wilcoxon p=0.0625 (n=5, min achievable).
  Spikes/gap n.s. Per decision table: "fuzzy ≈ fixed AUC" → null on accuracy;
  high-vol loss-variance hint needs more seeds before claiming stability win.

## Legacy files
- `legacy/data_loader.py`, `legacy/features.py`, `legacy/fuzzy_controller.py`,
  `legacy/evaluation.py` are legacy (old API, not imported by the pipeline)
  and kept for reference only.
  New pipeline: `config.py`, `data.py`, `fuzzy.py`, `model.py`, `train.py`,
  `evaluate.py`, `run_all.py`, `plots.py`.
  `main.py` is now a thin runner over `run_all.py`.
