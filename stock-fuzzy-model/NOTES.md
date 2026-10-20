# NOTES.md — gate log (gates T1..T8) + run results

## Code facts (no run needed)

- `END_DATE = "2025-12-31"` fixed; `WINDOW=30`; 5 stationary features;
  scaler + vol_rank fit/calibrated on train only.
- Model: 1-layer LSTM, 32 units, dropout 0.0, linear+sigmoid.
- Batching: contiguous, batch ORDER shuffled per epoch, partial batch kept.
  `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`, `GRAD_CLIP=1.0`.
- Active variants: `fixed_2e-4`, `fixed_5e-4`, `fuzzy`, `fuzzy_vol_only`.
  5 seeds x 4 = 20 runs.
- `BEST_FIXED` is chosen from the two fixed LRs by the lowest mean
  `best_val_loss` across seeds. Test metrics are not used for selection.
- Removed methods (fixed_mean_fuzzy, cosine, crisp, fuzzy_inverted,
  fuzzy_shuffled_vol) are no longer trained, registered, or reported.
  Old artifacts live under `outdated/` and `outdated/two_method/`.
- Two-input Mamdani controller (deterministic): surface mean 7.9313e-4
  (20.7% below 1e-3, inside the ±25% gate), range 1.2e-3,
  `max_step_vol` 5.78e-5, monotone non-increasing in `vol_rank` within a
  `2e-5` tolerance, Ruspini maxdiff 0.0, skfuzzy ControlSystem maxdiff
  7.16e-10 (< 1e-6). T4 shape checks PASS.
- Input MFs: `vol_rank` Ruspini Low/Med/High; `loss_trend` Falling
  `(-1,-1,0)`, Plateau `(-0.3,0,0.3)`, Rising `(0,1,1)`.
- Output MFs: Slow/Med/Fast triangles at `0.4e-3/1.0e-3/1.6e-3` over
  `[0, 2e-3]`; 9 rules; min implication, max aggregation, centroid.
- Prior Sugeno prototype kept in `fuzzy.py` as `SUGENO_CURVE` for
  historical reference only (not trained and not reported).

## Tautology notes (kept gates, honest scope)

- T4 surface gate verifies controller *shape* and bounds only.
- T6 reports the vol-only `corr(vol,lr)` and keeps it below -0.7 as a
  schedule check. The two-input `fuzzy` controller also moves LR with the
  trend, so its *global* corr is reported (-0.530 ± 0.176), not gated; its
  monotonicity is checked by `fuzzy.check_gate()` on the surface.
- Neither gate tests training benefit. Only T8 paired tests can.

## Run log (four-variant experiment, 2026-10-20)

- The locked reference run used the repository's cached `data/raw_GSPC.csv`
  after the live yfinance download timed out. The four-variant rerun
  re-downloaded the same date range and matched all data gates below.
- Data gates: T1 PASS (checksum 102657.2367); T2 PASS; T3 PASS
  (n_train=1893); T4 PASS; T5 PASS (batch_vol_std=0.248); T6 PASS
  (vol-only corr -0.996; fuzzy surface monotone); T7 PASS (no AUC>0.60);
  T8 done.
- Expected-vs-actual: 1893/382/383 windows; up-rates
  0.5404/0.5236/0.5770; checksum 102657.2367 — all MATCH.
- LR integrity: both fixed variants used exactly their configured LR
  at every step (std 0). `fuzzy` used 4.000e-4..1.600e-3 (mean
  1.320e-3); `fuzzy_vol_only` used 4.849e-4..1.536e-3 (mean 9.982e-4).
  Both stayed inside `[0, 2e-3]`.
- BEST_FIXED selection (validation loss only):
  `fixed_2e-4` = 0.692180; `fixed_5e-4` = 0.692329.
  Selected: **fixed_2e-4**.
- Results (test AUC mean ± std over 5 seeds):
  - `fixed_2e-4`: 0.5232 ± 0.0378
  - `fixed_5e-4`: 0.5326 ± 0.0230
  - `fuzzy`: 0.5343 ± 0.0089
  - `fuzzy_vol_only`: 0.5349 ± 0.0115
  - test accuracy is 0.5770 (majority) and MCC is 0.0 for every variant.
- Paired tests, `fuzzy` vs `fixed_2e-4`:
  - `test_auc`: p=0.4397 (Wilcoxon 0.6250) — no significant difference.
  - `best_val_loss`: p=0.1832 (0.1875).
  - `highvol_loss_var`: p=0.0013 (0.0625), but fuzzy is *higher/worse*
    (0.0004 vs 0.0003), so no stability win.
  - `train_val_gap`: p=0.7559 (0.8125).
- Paired tests, `fuzzy_vol_only` vs `fixed_2e-4`:
  - `test_auc`: p=0.3841 (0.6250) — no significant difference.
  - `best_val_loss`: p=0.0475 (0.1250).
  - `highvol_loss_var`: p=0.0019 (0.0625), again fuzzy is higher/worse.
- All 20 runs were constant predictors at the 0.5 threshold
  (pos_rate=1.0, tiny pred_std); val_loss ≈ ln2. This is an honest null:
  no claim that either fuzzy variant beats any fixed LR. Do not quote the
  archived results under `outdated/`.
