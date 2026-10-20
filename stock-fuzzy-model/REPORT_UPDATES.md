# REPORT_UPDATES.md — required report changes (actual run numbers only)

Source: `python prepare_data.py && python main.py`, 2026-10-20.
Never invent numbers; the values below are from `results.csv` and
`run_output.log`.

## RQ1 + Sec 2.5 + 3.3 + 4.4 + 6.2 (controller)

- The shipped controller is a Mamdani FIS: Ruspini Low/Med/High on
  `vol_rank`; overlapping Slow/Med/Fast triangles on LR `[1e-4, 3e-3]`;
  Low→Fast, Med→Med, High→Slow; min implication, max aggregation,
  centroid defuzzification; `skfuzzy.control` cross-check max diff `0.0`.
- Shipped centroids: Slow `0.4e-3`, Med `0.8e-3`, Fast `1.6e-3`.
  The requested `1.2e-3` / `2.4e-3` Med/Fast centroids produced a
  surface mean of `1.403e-3` (>20% above `1e-3`), so Med/Fast were scaled
  down. Shipped surface mean: `1.1334e-3` (within 20%).
- Surface range `1.3892e-3`; max step `3.0987e-5`; surface min
  `4.1075e-4`; surface max `1.8e-3`; Ruspini maxdiff `0.0`.
- The prior Sugeno prototype is not part of the current code.
- Only two methods are compared: `fixed_1e-3` (constant 0.001) and
  `fuzzy` (Mamdani, batch-volatility driven).
- Actual LR use: fixed min=max=mean=`1.000e-3`, std `0`;
  fuzzy range `5.042e-4 .. 1.758e-3`, mean `1.134e-3`, std `3.812e-4`.

## Sec 3.5 + 4.2 (model + training setup)

- Scaling: `StandardScaler` fit on TRAIN only, then clip to `[-3, 3]`.
- Features: 5 stationary price/volume features plus `vix_level` and
  `vix_z20` (7 total). VIX is same-day `^VIX` close, merged by date with
  no lookahead; it is cached in `data/raw_VIX.csv`.
- Model: 1-layer LSTM, 32 units, dropout 0.3 applied to the final hidden
  state before the linear classifier. (`nn.LSTM` internal dropout is 0
  for a single layer.)
- `WINDOW=20`, `BATCH_SIZE=32`, `EPOCHS=80`, `PATIENCE=15`,
  `MIN_EPOCHS=10`, `GRAD_CLIP=1.0`, Adam `weight_decay=1e-4`.
- Early stopping uses validation AUC (higher is better), never stops
  before epoch 10, and restores the best-AUC weights. `val_auc` is
  logged per epoch.
- Methods: 2 — `fixed_1e-3` and `fuzzy`. Regime batching is unchanged:
  contiguous batches, batch ORDER shuffled per epoch, partial final
  batch retained.

## Sec 5 / threshold

- The classification threshold is chosen on VALIDATION only, grid
  `0.40..0.60` step `0.005`, maximising balanced accuracy; the chosen
  threshold is applied unchanged to TEST and saved as `threshold_val`.
- Mean selected thresholds: fixed `0.5390`, fuzzy `0.5195`.

## Sec 6.1 (data: actuals vs report)

- Date lock: `END_DATE="2025-12-31"`.
- Windows: train/val/test `1900/391/392`.
- Up-rates: `0.5400 / 0.5269 / 0.5791`.
- T1 checksum: `-3942.92227483448` (matches expected).
- Test majority accuracy: `0.5791`.
- Figs 01–09 regenerated in `figs/`.

## Feature addition (validation-selected)

- Added `vix_level` and `vix_z20` (2 features).
- Selection used validation only. 10-seed validation AUC: `0.5480` for
  fuzzy versus `0.5147` before the features (+6.5% relative).
- Test was evaluated once after selection: fuzzy AUC `0.4941` versus
  `0.4741` before (+4.2% relative). The 5-7% target was met on validation
  but not on test.

## RQ2–RQ4 (actual results)

- `fixed_1e-3` (mean ± std over 10 seeds): test AUC `0.4930 ± 0.0099`,
  accuracy `0.4842 ± 0.0208`, balanced accuracy `0.5001 ± 0.0139`,
  MCC `0.0005 ± 0.0286`, best val loss `0.690972 ± 0.003477`,
  high-vol loss variance `0.002468 ± 0.001820`,
  best epoch `29.2 ± 20.49`.
- `fuzzy`: test AUC `0.4941 ± 0.0123`, accuracy `0.5000 ± 0.0208`,
  balanced accuracy `0.4867 ± 0.0215`, MCC `-0.0314 ± 0.0495`,
  best val loss `0.694628 ± 0.012723`, high-vol loss variance
  `0.002844 ± 0.002441`, best epoch `28.8 ± 20.95`.
- RQ2 (stability): `highvol_loss_var` paired-t `p=0.3131`, Wilcoxon
  `p=0.4316`. Fuzzy is higher (`0.002844` vs `0.002468`), so this is not
  a stability win.
- RQ3 (no AUC harm): paired-t `p=0.3804`, Wilcoxon `p=0.8457`.
- RQ4: `test_mcc` paired-t `p=0.0469`, Wilcoxon `p=0.0645` (fuzzy lower);
  `test_acc` paired-t `p=0.0334`, Wilcoxon `p=0.0156` (fuzzy higher);
  `best_val_loss` paired-t `p=0.2844`, Wilcoxon `p=0.6953`;
  `epochs_to_best` paired-t `p=0.8137`, Wilcoxon `p=1.0000`.

## Constant predictors + claim rule

- At the VAL-selected threshold: `0/20` runs are constant predictors.
- At the 0.5 threshold: `0/20` runs are constant.
- The model is not predictive: mean test AUC `0.4921` is below 0.5 and
  mean accuracy is below the majority baseline `0.5791`.
- No "fuzzy beats fixed" claim: AUC is null; accuracy favours fuzzy while
  MCC favours fixed. The VIX feature gain applies to both methods.
- T1..T8 summary: T1 PASS, T2 PASS, T3 PASS, T4 PASS, T5 PASS, T6 PASS,
  T7 PASS, T8 DONE. No gate FAIL.
