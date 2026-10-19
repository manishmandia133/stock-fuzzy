# NOTES.md — gate log (gates T1..T8; training numbers filled in by a run)

## Code facts (no run needed)

- `END_DATE = "2025-12-31"` fixed; `WINDOW=60`; 5 stationary features;
  scaler + vol_rank fit/calibrated on train only.
- Model: 2-layer LSTM, 64 units, dropout 0.2 between layers, linear+sigmoid.
- Batching: contiguous, batch ORDER shuffled per epoch, partial batch kept.
  `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`, `GRAD_CLIP=1.0`.
- 2 active methods: fixed_1e-3 (constant 0.001) / fuzzy (Mamdani LR from
  the actual training batch's mean vol_rank). 5 seeds x 2 = 10 runs.
- Removed methods (fixed_mean_fuzzy, cosine, crisp, fuzzy_inverted,
  fuzzy_shuffled_vol) are no longer trained, registered, or reported.
  Their old artifacts live under `outdated/` and are NOT current results.
- Mamdani controller (deterministic, verified from code): mean 1.00e-3
  (within 20% of 1e-3), range 1.2e-3, max step ~1.03e-5 (smooth),
  monotone decreasing, Ruspini maxdiff 0.0, skfuzzy ControlSystem max
  diff ~3.1e-17 (< 1e-9). T4 shape checks PASS.
- Prior Sugeno prototype kept in `fuzzy.py` as `SUGENO_CURVE` for
  historical reference only (not trained and not reported).

## Tautology notes (kept gates, honest scope)

- T4 curve gate + T6 corr(vol,lr)<-0.7 verify the controller *shape*/
  schedule application only — LR is a deterministic function of vol.
  They do not test training benefit. Only T8 paired tests can.

## Run log (two-method experiment, 2026-10-19)

- Data gates: T1 PASS (checksum 202000.3771); T2 PASS; T3 PASS
  (n_train=1863); T4 PASS; T5 PASS (batch_vol_std=0.251); T6 PASS
  (corr(vol,lr)=-0.996); T7 PASS (no AUC>0.60); T8 done.
- Expected-vs-actual: 1863/352/353 windows; up-rates 0.541/0.534/0.578;
  checksum 202000.3771 — all MATCH.
- LR integrity: `fixed_1e-3` used exactly 0.001 at every step
  (min=max=1.000e-3, std=0). `fuzzy` used 4.773e-4..1.527e-3
  (mean 9.991e-4), inside [0, 2e-3], corr(vol,lr)=-0.996.
- Results (mean ± std over 5 seeds):
  - `fixed_1e-3`: test AUC 0.5104 ± 0.0324, acc 0.5779, MCC 0.0,
    best val loss 0.69084 ± 0.00003.
  - `fuzzy`: test AUC 0.5062 ± 0.0341, acc 0.5779, MCC 0.0,
    best val loss 0.69085 ± 0.00004.
  - Paired fuzzy vs fixed: AUC p=0.104 (Wilcoxon 0.125); best val loss
    p=0.246 (0.188); high-vol loss variance p=0.008 (0.0625), with fuzzy
    slightly higher (0.000319 vs 0.000309).
- All 10 runs were constant predictors at the 0.5 threshold
  (pos_rate=1.0, pred_std≈1e-4); val_loss ≈ ln2. This is an honest null:
  no claim that fuzzy beats fixed. Do not quote the archived seven-variant
  results; they are under `outdated/`.
