# REPORT_UPDATES.md — exact report changes required by this code state

Numbers below come from a real full experiment run
(four variants x 5 seeds = 20 runs, 2026-10-20). Do not invent or reuse
archived values.

## RQ1 + Sec 2.5 + 3.3 + 4.4 + 6.2 (controller: two-input Mamdani)

- Replace the controller description: the shipped controller is a real
  **two-input Mamdani FIS**:
  - input 1 `vol_rank` with Ruspini Low/Med/High MFs (sum = 1);
  - input 2 `loss_trend` in `[-1, 1]` with Falling `(-1,-1,0)`, Plateau
    `(-0.3,0,0.3)`, Rising `(0,1,1)`;
  - output learning rate on `[0, 2e-3]` with Slow/Med/Fast triangles at
    `0.4e-3/1.0e-3/1.6e-3`;
  - 9 rules on (`vol_rank` x `loss_trend`) -> output; the exact table is
    `RULES` in `fuzzy.py` and is the source of truth; min implication,
    max aggregation, centroid defuzzification;
  - 101x101 lookup surface with bilinear interpolation; skfuzzy
    ControlSystem cross-check.
- `loss_trend` is
  `clip(((val_loss[-1] - mean(val_loss[-3:-1])) / (|mean| + 1e-8)) / 0.005, -1, 1)`;
  it is applied from the next epoch, and epochs 1-2 use `trend = 0`.
- New Mamdani facts (deterministic, verified from code, no run needed):
  mean 7.9313e-4 (within 25% of 1e-3), range 1.2e-3, max absolute step in
  `vol_rank` 5.78e-5, monotone non-increasing in `vol_rank` (2e-5
  tolerance), Ruspini maxdiff 0.0, skfuzzy maxdiff 7.16e-10.
- Keep the old Sugeno numbers explicitly as the **prior Sugeno prototype**
  (retained in `fuzzy.py` as `SUGENO_CURVE`, not trained or reported).
- Report `mean_lr_used`, `lr_min_used`, `lr_max_used`, `lr_std_used` per
  variant. RUN VALUES: `fuzzy` mean LR 1.320e-3 (4.000e-4..1.600e-3);
  `fuzzy_vol_only` mean 9.982e-4 (4.849e-4..1.536e-3); `vol_lr_corr`
  `fuzzy` -0.530 ± 0.176, `fuzzy_vol_only` -0.996.

## Sec 3.5 + 4.2 (model + training setup)

- Model: **1-layer** LSTM, 32 units, dropout **0.0**, linear + sigmoid.
  `model.py` is the source of truth.
- `WINDOW=30`, `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`, `GRAD_CLIP=1.0`.
- `drop_last` removed: the final partial batch is included.
- Variants: **4** — two fixed LRs (`2e-4`, `5e-4`) plus `fuzzy`
  (two-input Mamdani) and `fuzzy_vol_only` (one-input ablation). All other
  methods removed from the active experiment.
- `BEST_FIXED` is selected by lowest mean `best_val_loss` (validation
  only); report both fixed LRs: `fixed_2e-4` 0.692180, `fixed_5e-4`
  0.692329. Selected `BEST_FIXED = fixed_2e-4`.
- Regime batching unchanged: contiguous, batch ORDER shuffled per epoch.

## Sec 6.1 (data: actuals vs report)

- Date lock: `END_DATE = "2025-12-31"` fixed so these stay comparable.
- Console prints expected-vs-actual per item, plainly flagged. RUN RESULT:
  windows train/val/test = 1893/382/383; up-rates
  0.5404/0.5236/0.5770; T1 checksum 102657.2367. All MATCH.
- Figs 01-09 regenerated (`figs/`), including the new two-input MF figure
  (01), LR surface heatmap with slices (02), split balance (08), and
  T3/T4 measured-vs-threshold margins (09).

## RQ2-RQ4 (answerable only after the run)

- RQ2 (stability): `highvol_loss_var` paired test, fuzzy vs BEST_FIXED
  (`fixed_2e-4`). RUN RESULT: p=0.0013 (t), 0.0625 (Wilcoxon), but fuzzy
  is *higher/worse* (0.0004 vs 0.0003); `fuzzy_vol_only` p=0.0019
  (0.0625), also higher. No stability win.
- RQ3 (no AUC harm): `test_auc` paired tests + bootstrap CIs.
  RUN RESULT: `fuzzy` 0.5343 ± 0.0089 vs `fixed_2e-4` 0.5232 ± 0.0378,
  p=0.4397 (0.6250); `fuzzy_vol_only` 0.5349 ± 0.0115, p=0.3841
  (0.6250). No significant difference either way.
- All 20 runs were constant predictors (val_loss ≈ ln2; test accuracy =
  0.5770 majority; MCC = 0.0). Report the null honestly.
- Claim rule: no "fuzzy beats fixed" unless paired tests support it
  (p-values uncorrected, n=5, Wilcoxon floor 0.0625). A null is a null.
- Also report: constant-predictor count (20/20), val_loss≈ln2 note,
  T1..T8 summary table, and every gate result.
