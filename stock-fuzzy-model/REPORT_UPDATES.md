# REPORT_UPDATES.md — feature expansion and fuzzy-only pipeline

Source: actual `results.csv` generated 2026-10-21 by `python main.py`. Historical results are archived under `archive/2026-10-21_pre_feature_update/`.

## Changes

1. Added nine paper inputs plus eleven engineered inputs (20 total): returns/momentum, realized volatility, and SMA-trend features.
2. Target is next-day UP/DOWN only; all features are trailing and monthly macro series are release-lagged.
3. Removed fixed-LR variants from the active pipeline. Only the existing Mamdani fuzzy controller (`fuzzy`) is trained; `INITIAL_LR=0.001` is the optimizer initialisation, not an alternative mode.
4. Added precision, recall, F1, MCC, majority baseline, and confusion-matrix reporting.
5. Added a paired `full` vs `paper` feature ablation, both with fuzzy LR only.

## Features

- Paper (9): Close, VIX, EFFR, UNRATE, UMCSENT, USDX, MACD, ATR, RSI.
- Engineered (11): Return_1D, Return_2D, Return_5D, Return_10D, Return_20D, Volatility_5D, Volatility_20D, Volatility_Ratio, Close_vs_SMA_5, Close_vs_SMA_20, SMA_5_vs_SMA_20.
- `Return_20D` reconciles the prompt's 9+11=20 count with its enumerated 10 engineered features.

## Results

| feature set | AUC | accuracy | balanced acc | precision | recall | F1 | MCC | majority acc |
|---|---|---|---|---|---|---|---|---|
| paper | 0.5168 ± 0.0128 | 0.4793 ± 0.0370 | 0.5043 ± 0.0122 | 0.5829 ± 0.0304 | 0.3458 ± 0.2084 | 0.3971 ± 0.1803 | 0.0091 ± 0.0275 | 0.5791 |
| full | 0.5103 ± 0.0188 | 0.4825 ± 0.0432 | 0.5056 ± 0.0135 | 0.6016 ± 0.0466 | 0.3596 ± 0.2810 | 0.3858 ± 0.2122 | 0.0158 ± 0.0369 | 0.5791 |

- `paper` confusion (summed): [[3281, 1669], [4455, 2355]].
- `full` confusion (summed): [[3225, 1725], [4361, 2449]].

Feature ablation (`full` - `paper`) on `test_auc`: mean diff -0.00644, 95% CI [-0.01579, 0.00257], paired-t p=0.1915, Wilcoxon p=0.1840. Descriptive only; no superiority claim.

## Verification performed

T1 data reproducibility, T2 label/split checks, T3 shape/NaN checks, T4 controller curve, T5 batch-vol gate, T6 LR schedule bounds/non-zero variance, T7 leakage flag, and T8 feature contract all ran in `main.py`.

T7 leakage flags: 0. See `results.csv` and `NOTES.md`.

Historical fixed-LR files: `archive/2026-10-21_pre_feature_update/`.