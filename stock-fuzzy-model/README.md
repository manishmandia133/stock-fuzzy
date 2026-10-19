# Stock Fuzzy-LSTM Model

Volatility-aware fuzzy learning-rate control for an LSTM predicting
**^GSPC next-day direction** (`Close[t+1] > Close[t]`). Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # ^GSPC 2015-01-01..2025-12-31 -> data/*.csv
python main.py           # = run_all.py: 5 seeds x 2 methods -> results.csv + figs/
```

Active methods (10 runs total):

1. `fixed_1e-3` — constant learning rate `0.001` (baseline).
2. `fuzzy` — Mamdani fuzzy learning rate driven by the actual training
   batch's volatility (proposed method).

`END_DATE = "2025-12-31"` is fixed — report numbers are tied to it
(yfinance `end` is exclusive, `auto_adjust=True`).

## Files

- `config.py` — universe, dates, `WINDOW=60`, `BATCH_SIZE=32`, `EPOCHS=40`,
  `PATIENCE=6`, `GRAD_CLIP=1.0`, feature/target columns. Torch lazy:
  data/fuzzy scripts run without it.
- `prepare_data.py` — raw CSV as returned (flattened) + processed CSV
  (OHLCV, 5 features, target, 70/15/15 split, `*_scaled` fit on train only,
  `vol_rank` vs TRAIN vol, `fuzzy_lr`).
- `data.py` — download / stationary features / chronological split-scale
  (T1–T3 asserts inside) / window build sorted by end_date.
- `fuzzy.py` — **Mamdani FIS** (main): Ruspini Low/Med/High on vol_rank,
  Slow/Med/Fast triangles on LR `[0, 2e-3]` (centroids 0.4/1.0/1.6e-3),
  Low→Fast / Med→Med / High→Slow, min/max/centroid, 201-pt lookup,
  skfuzzy cross-check (<1e-9). The original Sugeno curve is retained in
  the module for historical reference only (not trained or reported).
- `model.py` — `DirectionLSTM`: **2 layers**, 64 units, dropout 0.2
  between layers, linear + sigmoid.
- `train.py` — regime-aware batching (contiguous, batch ORDER shuffled,
  final partial batch included). **2 methods**: `fixed_1e-3` (constant
  `0.001`) and `fuzzy` (Mamdani LR from this batch's mean vol_rank).
  Validates every optimizer step: fixed never changes; fuzzy stays within
  `[LR_MIN, LR_MAX]`. Records `mean_lr_used`, `lr_min_used`,
  `lr_max_used`, `lr_std_used`, and the relevant run configuration.
- `evaluate.py` — AUC + bootstrap 95% CI, acc, MCC, majority baseline,
  `pred_pos_rate`/`pred_std` + constant-predictor flag,
  `train_val_gap = val - train`, high-vol loss variance, leakage flag
  (AUC > 0.60), LogReg baseline.
- `run_all.py` — T1..T8 gate table, report expected-vs-actual, per-method
  mean±std, AUC CIs, LR-integrity checks, paired t + Wilcoxon
  (`fuzzy` vs `fixed_1e-3`, uncorrected, n=5, Wilcoxon floor 0.0625),
  and `run_config.json` with seeds/settings.
- `plots.py` — `figs/01..09` (MFs, curve + worked example at 0.62,
  price/vol/LR, LR-vs-vol, LR-over-steps, train+val loss with ln2,
  AUC boxplot, split balance, gate margins).

## Hypothesis

Fuzzy LR is more **stable** (especially high-vol loss variance) than the
fixed `0.001` baseline without hurting AUC. Null results are reported as
null — see NOTES.md and REPORT_UPDATES.md after a run.

## Outdated results

`outdated/results_7variants.csv` and `outdated/logs/` are from the previous
seven-variant experiment and are kept only for history. They are **not**
results of the current two-method experiment.
