# Stock Fuzzy-LSTM Model

Fuzzy-controlled learning-rate LSTM for **^GSPC next-day direction**
(`Close[t+1] > Close[t]`, binary).

## Pipeline

```bash
pip install -r requirements.txt
python prepare_data.py   # fetch ^GSPC -> data/raw_GSPC.csv + data/processed_GSPC.csv
python main.py           # = run_all.py: train 5 seeds x 5 variants -> results.csv + figs/
```

- `config.py` — universe (`^GSPC`), dates (`2015-01-01..2026-10-01`,
  yfinance `end` exclusive), window (60), hyper-params
  (`BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`), feature/target columns.
  Torch is imported lazily, so data/fuzzy scripts run without it.
- `prepare_data.py` — yfinance fetch (`auto_adjust=True`), raw CSV saved
  exactly as returned (MultiIndex flattened); processed CSV with OHLCV,
  `return_1d`, `hl_range`, `oc_ret`, `vol_chg`, `volatility_14d`,
  `target_direction`, chronological 70/15/15 `split`, `*_scaled`
  (MinMax fit on train only), `vol_rank` (percentile vs TRAIN vol
  distribution), `fuzzy_lr`. Prints head/tail/shape/NaN/per-split up-rate.
- `data.py` — `batch_download` / `engineer_features` / `split_scale_window`
  (chronological split, scaler + vol_rank calibrated on train only,
  label spot-check) / `build_dataset` (windows sorted by end_date,
  regime-aware order).
- `fuzzy.py` — real Mamdani FIS: vol_rank Low/Medium/High on a Ruspini
  triangular partition (memberships sum to 1); LR universe `[0, 2e-3]` with
  Slow/Medium/Fast triangles (centroids 0.4e-3 / 1.0e-3 / 1.6e-3);
  rules Low→Fast, Medium→Medium, High→Slow; min implication, max
  aggregation, centroid defuzzification. 201-pt lookup table for training,
  cross-checked against `skfuzzy.control.ControlSystem` (max diff < 1e-9).
- `model.py` — `DirectionLSTM` (1 layer, 64 units, dropout 0.2).
- `train.py` — regime-aware batching (contiguous batches, batch ORDER
  shuffled per epoch, final partial batch **included**). 5 variants:
  `fuzzy`, `fixed_1e-3`, `fixed_mean_fuzzy`, `fuzzy_inverted`,
  `fuzzy_shuffled_vol` (same empirical LR distribution, vol decoupled).
- `evaluate.py` — test AUC (with bootstrap 95% CI), acc, MCC, majority
  baseline, `pred_pos_rate` + `pred_std` with constant-predictor flag,
  `train_val_gap = val_loss - train_loss`, high-vol loss variance,
  leakage flag (AUC > 0.60), LogReg-on-last-day sanity AUC.
- `run_all.py` — prints fuzzy gate, split stats, per-variant mean±std,
  per-seed AUC CIs, constant-predictor summary, paired t + Wilcoxon
  (fuzzy vs `fixed_1e-3` / `fixed_mean_fuzzy` / `fuzzy_shuffled_vol` on
  `test_auc`, `best_val_loss`, `highvol_loss_var`; p-values **uncorrected**).
- `plots.py` — `figs/01_membership_input` … `07_auc_boxplot`
  (input/output MFs, control curve, worked inference example,
  price/vol/fuzzy-LR with split shading, train+val loss curves with ln2
  line, AUC boxplot) plus `08_lr_vs_vol` / `09_lr_over_step`
  (x-axis is optimizer step = shuffled batch order, not time).

## Honest result (2026-10-10 run, 5 seeds x 5 variants)

- Fuzzy gate: PASS (smooth, monotone decreasing, mean exactly 1e-3;
  Ruspini exact; skfuzzy max diff ≈ 3.1e-17).
- All 25 runs: val_loss ≈ ln2 (0.684–0.687) and **constant predictors**
  (`test_acc == majority`, MCC = 0.0). No usable 0.5-threshold edge.
- Test AUC ≈ 0.49 for every variant; paired tests n.s. (see NOTES.md).
  The corr(vol,lr) / curve gates only verify the controller shape —
  they do not test training benefit.
