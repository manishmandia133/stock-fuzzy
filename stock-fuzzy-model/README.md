# Stock Fuzzy-LSTM Model

Volatility-aware fuzzy learning-rate control for an LSTM predicting
**^GSPC next-day direction** (`Close[t+1] > Close[t]`). Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # ^GSPC 2015-01-01..2025-12-31 -> data/*.csv
python main.py           # = run_all.py: 10 seeds x 2 methods -> results.csv + figs/
```

Active methods (20 runs total):

1. `fixed_1e-3` — constant learning rate `0.001` (baseline).
2. `fuzzy` — Mamdani fuzzy learning rate driven by the actual training
   batch's volatility (proposed method).

`END_DATE = "2025-12-31"` is fixed — report numbers are tied to it
(yfinance `end` is exclusive, `auto_adjust=True`).

## Current setup

- Data: `StandardScaler` fit on TRAIN only, then `np.clip(x, -3, 3)`;
  chronological 70/15/15 split; `WINDOW=20`.
- Features: 5 stationary price/volume features plus `vix_level` (same-day
  `^VIX` close) and `vix_z20` (VIX 20-day z-score). VIX is merged by
  date with no lookahead and cached in `data/raw_VIX.csv`.
- Model: 1-layer LSTM, 32 units, dropout 0.3 on the final hidden state,
  linear + sigmoid.
- Training: Adam (`weight_decay=1e-4`), `BATCH_SIZE=32`, `EPOCHS=80`,
  `PATIENCE=15`, `MIN_EPOCHS=10`, `GRAD_CLIP=1.0`.
- Early stopping: validation AUC (maximised), best-AUC weights restored;
  `val_auc` logged per epoch.
- Threshold: selected on VALIDATION only (`0.40..0.60`, step `0.005`,
  maximum balanced accuracy), then applied unchanged to TEST.

## Files

- `config.py` — universe, dates, `WINDOW=20`, `BATCH_SIZE=32`,
  `EPOCHS=80`, `PATIENCE=15`, `MIN_EPOCHS=10`, `GRAD_CLIP=1.0`,
  `WEIGHT_DECAY=1e-4`, model sizes, threshold grid, feature/target
  columns. Torch is lazy: data/fuzzy scripts run without it.
- `prepare_data.py` — raw CSV as returned (flattened) + processed CSV
  (OHLCV + VIX, 7 features, target, 70/15/15 split, `*_scaled` using
  train-only StandardScaler + clip, `vol_rank` vs TRAIN vol, `fuzzy_lr`).
  Also writes the last 10 trading rows to `data/raw_GSPC_10d.csv` and
  `data/processed_GSPC_10d.csv`.
- `data.py` — download / VIX merge + stationary features / chronological
  split-scale (T1–T3 asserts inside) / window build sorted by end_date.
- `fuzzy.py` — **Mamdani FIS**: Ruspini Low/Med/High on vol_rank;
  overlapping Slow/Med/Fast triangles on LR `[1e-4, 3e-3]` with centroids
  `0.4e-3 / 0.8e-3 / 1.6e-3`; Low→Fast / Med→Med / High→Slow;
  min/max/exact centroid; 201-pt lookup; skfuzzy cross-check
  (max diff `0.0`). The requested `1.2e-3 / 2.4e-3` Med/Fast centroids
  put the surface mean at `1.403e-3` (>20% above `1e-3`), so they were
  scaled down; the shipped surface mean is `1.1334e-3`.
- `model.py` — `DirectionLSTM`: **1 layer**, 32 units, dropout 0.3
  before the linear classifier.
- `train.py` — regime-aware batching (contiguous, batch ORDER shuffled,
  final partial batch included). **2 methods**: `fixed_1e-3` (constant
  `0.001`) and `fuzzy` (Mamdani LR from this batch's mean vol_rank).
  Validates every optimizer step: fixed never changes; fuzzy stays within
  `[LR_MIN, LR_MAX]`. Records `mean_lr_used`, `lr_min_used`,
  `lr_max_used`, `lr_std_used`, and the relevant run configuration.
- `evaluate.py` — VAL-only threshold selection (balanced accuracy),
  AUC + bootstrap 95% CI, accuracy, balanced accuracy, MCC, majority
  baseline, `pred_pos_rate`/`pred_std` + constant-predictor flags,
  `train_val_gap = val - train`, high-vol loss variance, leakage flag
  (AUC > 0.60), LogReg baseline.
- `run_all.py` — T1..T8 gate table, report expected-vs-actual,
  per-method mean±std, AUC CIs, LR-integrity checks, paired t + Wilcoxon
  (`fuzzy` vs `fixed_1e-3`, uncorrected, n=10), constant-predictor
  counts, and `run_config.json` with seeds/settings.
- `plots.py` — `figs/01..09` (MFs, curve + worked example at 0.62,
  price/vol/LR, LR-vs-vol, LR-over-steps, train+val loss with ln2,
  AUC boxplot, split balance, gate margins).

## Current result (2026-10-20)

- `fixed_1e-3`: test AUC `0.4930 ± 0.0099`, accuracy `0.4842 ± 0.0208`,
  MCC `0.0005 ± 0.0286`.
- `fuzzy`: test AUC `0.4941 ± 0.0123`, accuracy `0.5000 ± 0.0208`,
  MCC `-0.0314 ± 0.0495`.
- Validation AUC: fixed `0.5461 ± 0.0167`, fuzzy `0.5480 ± 0.0171`.
  Adding the two VIX features raised fuzzy validation AUC by `6.5%`
  relative and test AUC by `4.2%` relative versus the previous
  five-feature run (`0.4741 -> 0.4941`).
- Majority accuracy: `0.5791`.
- Constant predictors: `0/20` at the VAL-selected threshold and `0/20`
  at 0.5. The original all-positive collapse is gone, but test AUC is
  still below 0.5 and accuracy is below majority.
- Paired tests: test AUC `p=0.3804` (t) / `0.8457` (Wilcoxon), so there
  is no fuzzy-vs-fixed AUC win. Accuracy is higher for fuzzy
  (`p=0.0334`), but MCC is lower for fuzzy (`p=0.0469`); do not claim a
  general fuzzy win. See `NOTES.md` and `REPORT_UPDATES.md`.

## Outdated results

`outdated/` contains earlier experiments and logs, including
`outdated/constant_collapse_multiseed/` (the previous 5-seed run),
`outdated/pre_vix_features/` (the five-feature baseline), and
`outdated/results_7variants.csv`. They are history only and are **not**
results of the current 10-seed two-method experiment.
