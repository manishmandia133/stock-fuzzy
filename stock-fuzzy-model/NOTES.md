# NOTES.md — gate log and run notes (T1..T8)

## Code facts (current state)

- End date fixed at `2025-12-31`; `WINDOW=20`.
- Features: 5 stationary price/volume features plus 2 VIX features:
  `vix_level` (same-day `^VIX` close) and `vix_z20` (VIX 20-day
  z-score). VIX is merged by date with no lookahead and cached in
  `data/raw_VIX.csv`. Total: 7 features.
- Scaling: `StandardScaler` fit on TRAIN only, then `np.clip(x, -3, 3)`.
- Model: 1-layer LSTM, 32 units, dropout 0.3 before the final linear layer.
- Training: Adam (`weight_decay=1e-4`), `BATCH_SIZE=32`, `EPOCHS=80`,
  `PATIENCE=15`, `MIN_EPOCHS=10`, `GRAD_CLIP=1.0`.
- Early stopping metric: validation AUC (higher is better); best-AUC
  weights are restored. Validation AUC is logged per epoch in
  `logs/<variant>_s<seed>_hist.csv`.
- Classification threshold: selected on VALIDATION only, grid
  `0.40..0.60` step `0.005`, maximising balanced accuracy; applied
  unchanged to TEST.
- 2 active methods: `fixed_1e-3` (constant 0.001) and `fuzzy` (Mamdani
  LR from the actual training batch's mean vol_rank). 10 seeds x 2 = 20
  runs.
- Mamdani controller: output universe `[1e-4, 3e-3]`; overlapping
  triangles with centroids 0.4e-3 / 0.8e-3 / 1.6e-3. The requested
  1.2e-3 / 2.4e-3 Med/Fast centroids made the surface mean 1.403e-3,
  outside the 20% band around 1e-3, so Med/Fast were scaled down. The
  shipped surface mean is 1.1334e-3.
- The prior Sugeno prototype is not part of the current code.

## Feature-selection note (no test tuning)

The two VIX features were selected on VALIDATION only. The selected set
(`vix_level` + `vix_z20`) reached 10-seed validation AUC `0.5480` versus
`0.5147` for the previous five-feature set (+6.5% relative). The test set
was evaluated once after this selection. Test AUC rose from `0.4741` to
`0.4941` (+4.2% relative), so the 5-7% target was met on validation but
not on test. Both methods use the same features, so the gain is not
specific to the fuzzy controller.

## Tautology notes (honest scope)

- T4 (curve shape) and T6 (corr(vol, lr) < -0.7) verify the controller
  and schedule application only. LR is a deterministic function of
  volatility. They do not test training benefit.
- Only T8 paired tests can support a claim that fuzzy beats fixed.

## Run log (2026-10-20)

`python prepare_data.py && python main.py` was run with 10 seeds x 2 methods.

### Data / integrity

- T1 PASS: checksum `-3942.92227483448` (same on two builds).
- T2 PASS: chronological split asserts + 3-window label spot-check.
- T3 PASS: `n_train=1900 > 1500`; no NaN.
- Split sizes: train 1900 / val 391 / test 392.
- Up-rates: train 0.5400 / val 0.5269 / test 0.5791.
- Test majority baseline: 0.5791.
- T5 PASS: `batch_vol_std=0.248` (> 0.15).
- T6 PASS: `corr(vol,lr)=-0.978` (< -0.7).
- T7 PASS: no test AUC > 0.60.
- T4 PASS: surface mean `1.1334e-3`; range `1.3892e-3`; max step
  `3.0987e-5`; surface min `4.1075e-4`; surface max `1.8e-3`; Ruspini
  maxdiff `0.0`; skfuzzy maxdiff `0.0`.

### Learning-rate integrity

- `fixed_1e-3`: exactly `1.000e-3` at every step (min=max, std=0).
- `fuzzy`: actual range `5.042e-4 .. 1.758e-3`; mean `1.134e-3`;
  std `3.812e-4`; inside `[1e-4, 3e-3]`.

### Results (mean ± std over 10 seeds)

- `fixed_1e-3`: test AUC `0.4930 ± 0.0099`; accuracy `0.4842 ± 0.0208`;
  balanced accuracy `0.5001 ± 0.0139`; MCC `0.0005 ± 0.0286`;
  best val loss `0.690972 ± 0.003477`; high-vol loss variance
  `0.002468 ± 0.001820`; best epoch `29.2 ± 20.49`.
- `fuzzy`: test AUC `0.4941 ± 0.0123`; accuracy `0.5000 ± 0.0208`;
  balanced accuracy `0.4867 ± 0.0215`; MCC `-0.0314 ± 0.0495`;
  best val loss `0.694628 ± 0.012723`; high-vol loss variance
  `0.002844 ± 0.002441`; best epoch `28.8 ± 20.95`.

### Constant-predictor outcome

- At the VAL-selected threshold: `0/20` runs are constant predictors.
- At the fixed 0.5 threshold: `0/20` runs are constant.
- The original all-runs-at-1.0 collapse is gone. The model remains
  non-predictive: mean test AUC is below 0.5 and mean accuracy `0.4921`
  is below the majority baseline `0.5791`.

### Paired tests (fuzzy vs fixed, n=10, uncorrected)

- `test_auc`: paired-t `p=0.3804`; Wilcoxon `p=0.8457`.
- `test_mcc`: paired-t `p=0.0469`; Wilcoxon `p=0.0645`.
  Fuzzy mean MCC is lower (`-0.0314` vs `0.0005`).
- `test_acc`: paired-t `p=0.0334`; Wilcoxon `p=0.0156`.
  Fuzzy mean accuracy is higher (`0.5000` vs `0.4842`).
- `best_val_loss`: paired-t `p=0.2844`; Wilcoxon `p=0.6953`.
- `highvol_loss_var`: paired-t `p=0.3131`; Wilcoxon `p=0.4316`.
- `epochs_to_best`: paired-t `p=0.8137`; Wilcoxon `p=1.0000`.

### Claim

No claim that fuzzy beats fixed. The AUC difference is null; the
accuracy difference favours fuzzy but the MCC difference favours fixed.
The VIX feature gain applies to both methods. Test AUC remains below 0.5
and accuracy remains below the majority baseline. Archived older
experiments under `outdated/` are not current results.
