# NOTES.md — protocol and gate log

> This file is rewritten by `python main.py` from the actual `results.csv`.
> Run the pipeline to populate the result numbers. Nothing below invents a
> run number.

## Protocol (code facts)

- Universe: `UNIVERSE_MODE="pooled"` — the notebook's 100-company source
  list (60 US + 40 India; 99 usable symbols, `TATAMOTORS.NS` delisted), or
  `"gspc"` for the original single-index experiment.
- Target: `TARGET_MODE="vol"` — label `1` if next-day `hl_range` is above
  the TRAIN-set median, else `0` (median fit on train only, aligned with
  window end `t`). `TARGET_MODE="direction"` remains available.
- There is one global set of chronological cut dates for all pooled
  tickers. `StandardScaler`, the vol_rank reference, and the vol-target
  median are fit on TRAIN only. Features are clipped to `[-3, 3]`.
- Model: 1-layer LSTM, 32 units, dropout 0.3 before the linear classifier.
- Training: Adam (`weight_decay=1e-4`), `WINDOW=20`, `EPOCHS=80`,
  `PATIENCE=15`, `MIN_EPOCHS=10`, `GRAD_CLIP=1.0`.
- Early stopping: 3-epoch moving-average validation loss. Best weights are
  restored only from `best_epoch >= 10`.
- Seeds: `range(30)`. Same seed gives the same initialisation and batch
  order for every variant (paired).
- Variants: `fixed_3e-4`, `fixed_1e-3`, `fixed_3e-3`, `fuzzy_dyn`,
  `fuzzy_vol`. No other scheduler is used.
- `fuzzy_dyn`: per-epoch Mamdani controller; `Improving/Worse/Flat` from
  relative val-loss change, `LowVol/HighVol` from epoch mean vol_rank;
  3x2 rules, min-AND, weighted-average defuzzification; LR multiplier in
  `[0.40, 1.15]`; `lr = clip(lr*factor, LR_MIN, LR_MAX)`; 3-epoch linear
  warmup; start LR `1e-3`.
- Best fixed LR is selected on **validation AUC only** (mean over seeds).
- Primary metric, fixed in advance: **test AUC**. Secondary: test log-loss,
  epochs-to-target val loss, area under the val-loss curve, high-vol test
  AUC (top tercile of vol_rank), across-seed std.
- Paired tests: bootstrap 95% CI for the mean difference, paired-t, and
  Wilcoxon. Holm correction applies to secondary metrics; the primary is
  reported uncorrected. Only the primary decides the claim.

## Tautology notes

- **T4** (fuzzy_vol curve shape) and **T6** (LR bounds / `corr(vol, lr)`)
  are tautological. They verify that the controller and schedule were
  applied as specified; they do not test training benefit.
- Only the paired primary-metric test can support a fuzzy-beats-fixed claim.

## Gate log

`python main.py` prints and writes the T1..T8 outcomes and all actual
numbers. T4/T6 remain labelled tautological. The claim is printed as
`NULL RESULT` unless the primary rule is satisfied.

## Run log

Run `python prepare_data.py && python main.py` to populate this section
with actual values from `results.csv`.
