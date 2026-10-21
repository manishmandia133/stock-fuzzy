# REPORT_UPDATES.md — required report changes

> This file is rewritten by `python main.py` from the actual `results.csv`.
> It intentionally contains no invented run numbers. Run the pipeline to
> populate the result tables and p-values.

## Changes implemented

1. **Target (`config.py`, `data.py`)**
   - Added `TARGET_MODE = "vol"` (default) and kept `"direction"`.
   - Vol label: `1` if next-day `hl_range` (configurable to `|return|`)
     exceeds the TRAIN-set median, else `0`. Median fit on train only;
     label aligned with window end `t`; T2 spot-checks updated for both
     targets and verify the next-day quantity.

2. **Data scope (`data.py`)**
   - Added `UNIVERSE_MODE = "pooled"`: the notebook's 100-company source
     list (60 US + 40 India; 99 usable, `TATAMOTORS.NS` delisted), windows
     built per ticker then concatenated.
   - One global set of chronological cut dates shared by all tickers.
   - Scaler, vol_rank reference, and vol-target median fit on TRAIN only.
   - `^GSPC`-only mode remains available.
   - Per-ticker raw data is cached so `prepare_data.py` and `main.py` do not
     download twice.

3. **Fuzzy controller (`fuzzy.py`)**
   - Existing Mamdani `vol_rank -> LR` curve kept as `fuzzy_vol` (ablation).
   - Added `fuzzy_dyn`: per-epoch controller using
     `dloss=(vloss-prev_vloss)/prev_vloss` and epoch mean vol_rank.
     Memberships and 3x2 rules exactly as specified; min-AND;
     weighted-average defuzzification; multiplier in `[0.40, 1.15]`;
     `lr = clip(lr*factor, LR_MIN, LR_MAX)`; start LR `1e-3`; 3-epoch
     linear warmup.
   - Added unit tests in `tests/test_fuzzy.py` for factor bounds,
     monotonicity, membership partition, warmup, and LR bounds.

4. **Baselines and training (`train.py`)**
   - Variants: `fixed_3e-4`, `fixed_1e-3`, `fixed_3e-3`, `fuzzy_dyn`,
     `fuzzy_vol`. No ReduceLROnPlateau, cosine decay, or other scheduler.
   - Best fixed LR selected on validation AUC only; paired comparison uses
     that choice.
   - Mean LR used is logged for every run.
   - Early stopping on 3-epoch moving-average validation loss; best weights
     restored only from `best_epoch >= 10`.
   - Same seed -> same init and batch order for all variants (paired);
     `SEEDS = range(30)`.

5. **Metrics and statistics (`evaluate.py`, `run_all.py`)**
   - Primary: test AUC. Secondary: test log-loss, epochs-to-target val
     loss, area under val-loss curve, high-vol test AUC, across-seed std.
   - Paired mean difference with 95% bootstrap CI, paired-t, Wilcoxon.
   - Holm correction across secondary metrics; primary uncorrected.
   - Claim rule: primary p < 0.05 AND CI excludes 0 AND direction favours
     fuzzy; otherwise `NULL RESULT`.

6. **Reporting (`plots.py`, `run_all.py`)**
   - Regenerated loss curves, LR-over-epoch for `fuzzy_dyn`, and AUC
     boxplot including all fixed LRs.
   - `NOTES.md` and `REPORT_UPDATES.md` are written from actual results at
     the end of `main.py`.
   - T1–T8 kept; T4/T6 labelled tautological.

## Results

Run `python prepare_data.py && python main.py`. The run overwrites this
section with actual numbers copied programmatically from `results.csv`.
