# Stock Fuzzy-LSTM Model

Volatility-aware fuzzy learning-rate control for an LSTM predicting
**^GSPC next-day direction** (`Close[t+1] > Close[t]`). Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # ^GSPC 2015-01-01..2025-12-31 -> data/*.csv
python main.py           # = run_all.py: 5 seeds x 7 variants -> results.csv + figs/
```

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
  skfuzzy cross-check (<1e-9). Plus the **original Sugeno curve** as a
  printed reference (not trained).
- `model.py` — `DirectionLSTM`: **2 layers**, 64 units, dropout 0.2
  between layers, linear + sigmoid.
- `train.py` — regime-aware batching (contiguous, batch ORDER shuffled,
  final partial batch included). **7 variants**: `fuzzy`, `fixed_1e-3`,
  `fixed_mean_fuzzy` (key comparator, same mean), `cosine`, `crisp`,
  `fuzzy_inverted`, `fuzzy_shuffled_vol`. Records `mean_lr_used` (RQ1).
- `evaluate.py` — AUC + bootstrap 95% CI, acc, MCC, majority baseline,
  `pred_pos_rate`/`pred_std` + constant-predictor flag,
  `train_val_gap = val - train`, high-vol loss variance, leakage flag
  (AUC > 0.60), LogReg baseline.
- `run_all.py` — T1..T8 gate table, report expected-vs-actual, per-variant
  mean±std, AUC CIs, `mean_lr_used`, paired t + Wilcoxon vs all 6 others
  (uncorrected, n=5, Wilcoxon floor 0.0625).
- `plots.py` — `figs/01..09` (MFs, curve + worked example at 0.62,
  price/vol/LR, LR-vs-vol, LR-over-steps, train+val loss with ln2,
  AUC boxplot, split balance, gate margins).

## Hypothesis

Fuzzy LR is more **stable** (esp. high-vol loss variance) than a fixed LR
**with the same mean**, without hurting AUC. Key comparator:
`fixed_mean_fuzzy`. RQ4 attribution: `cosine`, `crisp`, `fuzzy_inverted`,
`fuzzy_shuffled_vol`. Null results are reported as null — see NOTES.md
and REPORT_UPDATES.md after a run.
