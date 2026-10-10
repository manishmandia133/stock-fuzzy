# REPORT_UPDATES.md — exact report changes required by this code state

Fill every `PENDING RUN` item from a real
`python prepare_data.py && python main.py` run. Never invent numbers.

## RQ1 + Sec 2.5 + 3.3 + 4.4 + 6.2 (controller: Sugeno → Mamdani)

- Replace the controller description: the shipped controller is a real
  **Mamdani FIS** (Ruspini Low/Med/High on vol_rank; Slow/Med/Fast
  triangles on LR `[0, 2e-3]` with centroids 0.4e-3/1.0e-3/1.6e-3;
  Low→Fast, Med→Med, High→Slow; min implication, max aggregation,
  centroid defuzz; 201-pt lookup; skfuzzy cross-check max diff < 1e-9).
- Keep the old numbers explicitly as the **prior Sugeno prototype**
  (weighted average, trapmf[0,0,.25,.55] / trimf[.2,.5,.8] /
  trapmf[.45,.75,1,1], anchors 1.5/1.0/0.4e-3; mean ~9.63e-4, max step
  ~1.79e-5). `run_all.py` prints live Sugeno gate numbers for traceability.
- New Mamdani facts (deterministic, verified from code, no run needed):
  mean 1.00e-3, range 1.2e-3, max step ~1.03e-5, monotone, Ruspini exact.
- RQ1 fairness evidence: `mean_lr_used` per variant table (proves fuzzy vs
  fixed_mean_fuzzy ran on the same mean LR budget). PENDING RUN values.

## Sec 3.5 + 4.2 (model + training setup)

- Model: **2-layer** LSTM, 64 units, dropout **0.2 between layers**,
  linear + sigmoid (was 1-layer). `model.py` is the source of truth.
- `BATCH_SIZE=32` (was 128, ~14 steps/epoch, never learned),
  `EPOCHS=40`, `PATIENCE=6`, `GRAD_CLIP=1.0`.
- `drop_last` removed: final partial batch is now included (previously the
  newest ~71 train windows were silently dropped every epoch).
- Variants: **7** — fuzzy, fixed_1e-3, fixed_mean_fuzzy, cosine
  (CosineAnnealingLR from 1e-3 over epochs), crisp (vol_rank thirds →
  1.6/1.0/0.4e-3), fuzzy_inverted, fuzzy_shuffled_vol.
- Regime batching unchanged: contiguous, batch ORDER shuffled per epoch.

## Sec 6.1 (data: actuals vs report)

- Date lock: `END_DATE = "2025-12-31"` fixed so these stay comparable.
- Console prints expected-vs-actual per item, plainly flagged:
  windows train/val/test 1863/352/353; up-rates 0.541/0.534/0.578;
  T1 checksum 202000.3771. PENDING RUN: copy the MATCHES/DIFFERS lines.
- Figs 01–09 regenerated (`figs/`): 08 = split balance + window counts,
  09 = T3/T4 measured-vs-threshold margins.

## RQ2–RQ4 (answerable only after the run)

- RQ2 (stability): `highvol_loss_var` paired tests, fuzzy vs all 6.
- RQ3 (no AUC harm): `test_auc` paired tests + bootstrap CIs.
- RQ4 (attribution): cosine / crisp / inverted / shuffled columns.
- Claim rule: no "fuzzy beats fixed" unless paired tests support it
  (p-values uncorrected, n=5, Wilcoxon floor 0.0625). A null is a null.
- Also report: constant-predictor count, val_loss≈ln2 note if present,
  T1..T8 summary table, every gate FAIL listed.
