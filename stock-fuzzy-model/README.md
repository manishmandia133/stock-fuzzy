# Stock Fuzzy-LSTM Model

Volatility-aware fuzzy learning-rate control for an LSTM predicting
**^GSPC next-day direction** (`Close[t+1] > Close[t]`). Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # ^GSPC 2015-01-01..2025-12-31 -> data/*.csv
python main.py           # 5 seeds x 7 variants -> results.csv + figs/
```

Active variants (35 runs total):

1. `fixed_2e-4`
2. `fixed_5e-4`
3. `fixed_1e-3`
4. `fixed_1p5e-3`
5. `fixed_2e-3`
6. `fuzzy` — two-input Mamdani LR from (`vol_rank`, validation-loss trend)
7. `fuzzy_vol_only` — one-input ablation using only `vol_rank`

`BEST_FIXED` is selected from the fixed grid using **validation loss only**
(lowest mean `best_val_loss`); test metrics never influence selection.

`END_DATE = "2025-12-31"` is fixed — report numbers are tied to it
(yfinance `end` is exclusive, `auto_adjust=True`).

## Files

- `config.py` — universe, dates, `WINDOW=30`, `HIDDEN=32`, `NUM_LAYERS=1`,
  `DROPOUT=0.0`, `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`,
  `GRAD_CLIP=1.0`, `FIXED_LR_GRID`/`FIXED_LR_VARIANTS`, `TREND_SCALE`,
  feature/target columns. Torch lazy: data/fuzzy scripts run without it.
- `prepare_data.py` — raw CSV (flattened) + processed CSV (OHLCV, 5
  features, target, 70/15/15 split, `*_scaled` fit on train only,
  `vol_rank` vs TRAIN vol, `fuzzy_lr` vol-only reference).
- `data.py` — download / stationary features / chronological split-scale
  (T1–T3 asserts inside) / window build sorted by end_date.
- `fuzzy.py` — **two-input Mamdani FIS**: Ruspini Low/Med/High on
  `vol_rank`; Falling `(-1,-1,0)` / Plateau `(-0.3,0,0.3)` / Rising
  `(0,1,1)` on `loss_trend`; Slow/Med/Fast triangles on LR
  `[0, 2e-3]` (centroids 0.4/1.0/1.6e-3); 9 rules; min/max/centroid;
  101x101 lookup `LR_SURFACE`; bilinear `fuzzy_lr(vol, trend)`;
  vol-only `fuzzy_lr_vol_only(v)`; skfuzzy cross-check (<1e-6).
  `check_gate()` asserts bounds, surface monotonicity in `vol_rank`
  (tolerance `2e-5`), mean within 25% of `1e-3`, and Ruspini partition.
- `model.py` — `DirectionLSTM`: 1 layer, 32 units, dropout 0.0, linear +
  sigmoid; defaults read from `config.py`.
- `train.py` — regime-aware batching (contiguous, batch ORDER shuffled,
  final partial batch included). Seven variants: five constant LRs, plus
  `fuzzy`/`fuzzy_vol_only`. Fuzzy LR is computed per batch from the batch's
  mean `vol_rank` and the previous epoch's validation-loss trend
  (`trend=0` for epoch 1; updated each epoch). Validates every optimizer
  step: fixed variants never change; fuzzy stays within `[LR_MIN, LR_MAX]`.
  Records `mean_lr_used`, `lr_min_used`, `lr_max_used`, `lr_std_used`,
  trend stats, and the run configuration.
- `evaluate.py` — AUC + bootstrap 95% CI, acc, MCC, majority baseline,
  `pred_pos_rate`/`pred_std` + constant-predictor flag,
  `train_val_gap = val - train`, high-vol loss variance, leakage flag
  (AUC > 0.60), LogReg baseline.
- `run_all.py` — T1..T8 gate table, expected-vs-actual report, per-variant
  mean±std, AUC CIs, BEST_FIXED selection (validation only), LR-integrity
  checks, paired t + Wilcoxon (`fuzzy` and `fuzzy_vol_only` vs BEST_FIXED,
  uncorrected, n=5, Wilcoxon floor 0.0625), and `run_config.json`.
- `plots.py` — `figs/01..09` (input MFs for vol and trend + output MFs,
  LR surface heatmap with slices, price/vol/vol-only LR, LR-vs-vol
  colored by trend, LR/vol/trend over steps, train+val loss with ln2,
  AUC boxplot, split balance, gate margins).

## Hypothesis

Fuzzy LR would be more **stable** (especially high-vol loss variance) than
a fixed baseline without hurting AUC. On the current data the hypothesis is
not supported: all 35 runs are constant predictors and paired tests show no
AUC gain. Null results are reported as null — see NOTES.md and
REPORT_UPDATES.md.

## Outdated results

`outdated/` holds the previous seven-variant experiment. `outdated/two_method/`
holds the later two-method experiment. Neither is part of the active
experiment; do not quote them as current results.
