# NOTES.md — gate log (gates T1..T8; training numbers filled in by a run)

## Code facts (no run needed)

- `END_DATE = "2025-12-31"` fixed; `WINDOW=60`; 5 stationary features;
  scaler + vol_rank fit/calibrated on train only.
- Model: 2-layer LSTM, 64 units, dropout 0.2 between layers, linear+sigmoid.
- Batching: contiguous, batch ORDER shuffled per epoch, partial batch kept.
  `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`, `GRAD_CLIP=1.0`.
- 7 variants: fuzzy / fixed_1e-3 / fixed_mean_fuzzy / cosine / crisp /
  fuzzy_inverted / fuzzy_shuffled_vol.
- Mamdani controller (deterministic, verified from code): mean 1.00e-3
  (within 20% of 1e-3), range 1.2e-3, max step ~1.03e-5 (smooth),
  monotone decreasing, Ruspini maxdiff 0.0, skfuzzy ControlSystem max
  diff ~3.1e-17 (< 1e-9). T4 shape checks PASS.
- Prior Sugeno prototype kept as `SUGENO_CURVE` reference (not trained);
  `run_all.py` prints its live gate numbers next to the report's
  (mean ~9.63e-4, max step ~1.79e-5) for traceability.

## Tautology notes (kept gates, honest scope)

- T4 curve gate + T6 corr(vol,lr)<-0.7 verify the controller *shape*/
  schedule application only — LR is a deterministic function of vol.
  They do not test training benefit. Only T8 paired tests can.

## Run log

- No training run has been executed against this exact code state yet
  (2-layer model + 7 variants + 2025-12-31 date). Run
  `python prepare_data.py && python main.py`, then record here:
  T1..T8 PASS/FAIL, expected-vs-actual (1863/352/353; 0.541/0.534/0.578;
  checksum 202000.3771), per-variant mean±std, AUC CIs, mean_lr_used,
  paired p-values, constant-predictor count.
- Prior run (different code: 1-layer, 5 variants, 2026-10-01) was an
  honest null — val_loss ≈ ln2, 25/25 constant predictors. It does NOT
  transfer to this code state; do not quote its numbers as results.
