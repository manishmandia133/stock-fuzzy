# NOTES.md — protocol, feature contract, and actual run numbers

Generated from `results.csv` on 2026-10-21T14:29:48.065084+00:00.

## Protocol (code facts)

- Universe mode: `gspc`; contributing tickers: paper=1, full=1 (requested 1).
- Target: next-day UP/DOWN, `target_direction[t] = 1[Close[t+1] > Close[t]]`. No volatility target is active.
- Chronological split; StandardScaler fit on TRAIN only; clip to [-3, 3]. No shuffling across time.
- Window 20; batch size 32; max epochs 80; patience 15.
- Early stopping: 3-epoch moving-average validation loss; restore best weights with best_epoch >= 10.
- Seeds: 0..29 (30 seeds).
- Active method: `fuzzy` only. The optimizer starts at `INITIAL_LR=0.001` and the Mamdani controller sets LR from each training batch's vol_rank.
- No fixed-LR variant, ReduceLROnPlateau, cosine decay, or other scheduler is active.
- Feature sets: `paper` = 9 inputs; `full` = 20 inputs.
- `full` = nine paper features + eleven engineered features. The engineered list is 4 returns + `Return_20D` + 3 volatility + 3 trend features. `Return_20D` is included to reconcile the prompt's 9+11=20 count with its enumerated 10 engineered features; it uses the same close-only trailing construction.
- Features are all trailing: returns, rolling std, SMA, EMA, ATR, and RSI use data at or before t. Monthly FRED series (UNRATE, UMCSENT) are shifted one month before daily ffill.
- Test metrics are computed once per trained model. The classification threshold is selected on VALIDATION only.

## Feature contract

- `paper` (9): Close, VIX, EFFR, UNRATE, UMCSENT, USDX, MACD, ATR, RSI
- `full` (20): Close, VIX, EFFR, UNRATE, UMCSENT, USDX, MACD, ATR, RSI, Return_1D, Return_2D, Return_5D, Return_10D, Return_20D, Volatility_5D, Volatility_20D, Volatility_Ratio, Close_vs_SMA_5, Close_vs_SMA_20, SMA_5_vs_SMA_20

## Results (mean ± std over seeds, VAL-selected threshold)

| feature set | AUC | accuracy | balanced acc | precision | recall | F1 | MCC | majority acc | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| paper | 0.5168 ± 0.0128 | 0.4793 ± 0.0370 | 0.5043 ± 0.0122 | 0.5829 ± 0.0304 | 0.3458 ± 0.2084 | 0.3971 ± 0.1803 | 0.0091 ± 0.0275 | 0.5791 | 0.6944 ± 0.0045 |
| full | 0.5103 ± 0.0188 | 0.4825 ± 0.0432 | 0.5056 ± 0.0135 | 0.6016 ± 0.0466 | 0.3596 ± 0.2810 | 0.3858 ± 0.2122 | 0.0158 ± 0.0369 | 0.5791 | 0.6983 ± 0.0078 |

- `paper` summed test confusion matrix: [[3281, 1669], [4455, 2355]] (TN, FP / FN, TP).
- `paper` AUC undefined runs: 0.
- `paper` mean LR used: 1.1324e-03; LR range [5.1173e-04, 1.7807e-03].
- `paper` early-stop epochs: mean 13.17, range [10, 33].
- `full` summed test confusion matrix: [[3225, 1725], [4361, 2449]] (TN, FP / FN, TP).
- `full` AUC undefined runs: 0.
- `full` mean LR used: 1.1324e-03; LR range [5.1173e-04, 1.7807e-03].
- `full` early-stop epochs: mean 12.63, range [10, 23].

## Feature ablation: `full` minus `paper` (paired by seed)

| metric | mean diff | 95% bootstrap CI | paired-t p | Wilcoxon p | Holm p (secondary) |
|---|---|---|---|---|---|
| test_auc | -0.00644 | [-0.01579, 0.00257] | 0.1915 | 0.1840 | — |
| test_logloss | 0.00391 | [0.00092, 0.00684] | 0.0164 | 0.0040 | 0.1641 |
| test_acc | 0.00323 | [-0.01369, 0.02032] | 0.7177 | 0.6361 | 1.0000 |
| test_bal_acc | 0.00125 | [-0.00534, 0.00792] | 0.7240 | 0.8394 | 1.0000 |
| test_precision | 0.01875 | [-0.00417, 0.04087] | 0.1215 | 0.1048 | 0.9717 |
| test_recall | 0.01380 | [-0.08899, 0.12056] | 0.8014 | 0.8712 | 1.0000 |
| test_f1 | -0.01131 | [-0.09962, 0.07835] | 0.8065 | 0.8553 | 1.0000 |
| test_mcc | 0.00664 | [-0.01106, 0.02430] | 0.4791 | 0.4280 | 1.0000 |
| test_auc_highvol | 0.01544 | [0.00046, 0.03012] | 0.0588 | 0.0606 | 0.5288 |
| epochs_to_target_val_loss | 1.16667 | [-3.96667, 6.10000] | 0.6577 | 0.5375 | 1.0000 |
| aulc_val_loss | -0.28160 | [-1.87095, 1.10006] | 0.7172 | 1.0000 | 1.0000 |

The ablation is descriptive. No feature-superiority claim is made: the primary metric is reported uncorrected and the secondary metrics are Holm-corrected.

## Gates

- T1 data reproducibility: PASS.
- T2 chronological split + label spot-checks: PASS.
- T3 window shapes/NaN: PASS.
- T4 fuzzy controller curve: PASS (shape only; mean LR 1.1334e-03).
- T5 regime batching: all runs batch_vol_std > 0.15.
- T6 fuzzy schedule applied: all runs LR in bounds with non-zero LR variance.
- T7 leakage flag: PASS (0 run(s) with test AUC > 0.60).
- T8 feature contract: PASS (paper=9, full=20, shape checks).
- Historical fixed-LR artifacts: `archive/2026-10-21_pre_feature_update/`.
