# Stock Fuzzy-LSTM Model

Fuzzy learning-rate control for an LSTM predicting next-day **UP/DOWN**
(`Close[t+1] > Close[t]`). Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # cache market/macro data + ^GSPC inspection CSV
python main.py           # 30 seeds x {paper, full} feature sets, fuzzy LR only
```

Useful limited runs:

```bash
python main.py --seeds 0 1 2 --feature-sets paper full
python main.py --universe gspc --seeds 0 1 2
```

`main.py` writes `results.csv`, `run_config.json`, `NOTES.md`,
`REPORT_UPDATES.md`, `logs/`, and `figs/` from the actual run. Numbers are
never hand-entered.

## Target

`target_direction[t] = 1` if `Close[t+1] > Close[t]`, else `0`.
The target is next-day direction only; no volatility target is active.

## Input features (20)

### Nine paper features

`Close`, `VIX`, `EFFR`, `UNRATE`, `UMCSENT`, `USDX`, `MACD`, `ATR`, `RSI`.

- `EFFR`, `UNRATE`, `UMCSENT` come from FRED and are cached.
- `USDX` is the ICE US Dollar Index (`DX-Y.NYB`); `VIX` is `^VIX`.
- `MACD = EMA12 - EMA26`, `ATR` is the 14-day average true range, and `RSI`
  is the 14-day Wilder RSI.

### Eleven engineered features

Returns/momentum (5): `Return_1D`, `Return_2D`, `Return_5D`, `Return_10D`,
`Return_20D`.

Realized volatility (3): `Volatility_5D`, `Volatility_20D`,
`Volatility_Ratio = Volatility_5D / Volatility_20D` (safe division; zero or
non-finite denominators become NaN and are dropped).

Moving-average trends (3): `Close_vs_SMA_5 = Close/SMA_5 - 1`,
`Close_vs_SMA_20 = Close/SMA_20 - 1`,
`SMA_5_vs_SMA_20 = SMA_5/SMA_20 - 1`.

The prompt enumerates ten engineered features but asks for eleven and a
20-input total. `Return_20D` is the eleventh feature; it uses the same
close-only, trailing construction as the other returns. The code checks the
feature contract explicitly (`paper=9`, `full=20`).

## Leakage controls

- Chronological 70/15/15 split; no random shuffle across time.
- All rolling/EMA features are trailing (`t` and earlier). No centered windows.
- `StandardScaler` is fit on TRAIN only, then applied unchanged to val/test;
  transformed features are clipped to `[-3, 3]`.
- `vol_rank` is ranked against the TRAIN volatility distribution only.
- Monthly FRED series (`UNRATE`, `UMCSENT`) are shifted one month before the
  daily forward-fill to respect release lags. `EFFR` is daily; holidays are
  forward-filled.
- The target at row `t` uses `Close[t+1]`; input windows end at `t`.
- Rolling-window NaNs are dropped before window construction, so labels stay
  aligned with the original dates. Label spot-checks assert the next close.

## Learning rate: fuzzy only

There is **one active method**: `fuzzy`, the existing Mamdani controller that
maps the current training batch's `vol_rank` to a learning rate.

- Optimizer family: Adam (`weight_decay=1e-4`).
- `INITIAL_LR = 1e-3` is only the optimizer's starting value; the controller
  sets the LR from the first training batch onward.
- LR range `[1e-4, 3e-3]`; the controller curve is Low→Fast, Medium→Medium,
  High→Slow.
- Every optimizer step logs `vol_rank`, selected LR, and batch loss; every
  epoch logs mean LR, train loss, validation loss, validation accuracy/AUC.
- No fixed-LR variant, ReduceLROnPlateau, cosine decay, or other scheduler is
  active.
- The per-epoch `fuzzy_dyn` experiment from the previous task is archived and
  is not part of the active pipeline.

## Model and training

- 1-layer LSTM, 32 hidden units, dropout 0.3 before the linear classifier.
- `WINDOW=20`, `EPOCHS=80`, `PATIENCE=15`, `MIN_EPOCHS=10`,
  `GRAD_CLIP=1.0`.
- Regime-aware batching: contiguous windows sorted by end date; batch order is
  permuted per epoch, never samples across batches.
- Early stopping on the 3-epoch moving-average validation loss; best eligible
  weights (`best_epoch >= 10`) are restored.
- Same seeds for `paper` and `full`, so the feature ablation is paired by seed.
- Seeds: `range(30)`.

## Evaluation

- Threshold selected on VALIDATION only (balanced accuracy over
  `0.40..0.60`, step `0.005`), then applied unchanged to TEST.
- Reported at the selected threshold: ROC-AUC, accuracy, balanced accuracy,
  precision, recall, F1, MCC, majority-class accuracy, confusion matrix.
- AUC is reported as `NaN` when the test split has only one class; bootstrap
  AUC CIs are percentile intervals.
- The `full` vs `paper` ablation is paired by seed (bootstrap CI, paired-t,
  Wilcoxon; Holm correction on secondary metrics). It is descriptive and makes
  no feature-superiority claim.

## Files

- `config.py` — feature sets/contract, target, universe, model/training
  settings, fuzzy LR constants.
- `data.py` — downloads/caches OHLCV + VIX/USDX/FRED, feature engineering,
  train-only split/scaler, window construction, label checks.
- `fuzzy.py` — Mamdani `vol_rank -> LR` controller (active); the archived
  `fuzzy_dyn` code remains for the unit tests.
- `model.py` — `DirectionLSTM`.
- `train.py` — fuzzy-only training loop, LR logging, early stopping.
- `evaluate.py` — metrics, confusion matrix, bootstrap/paired statistics.
- `run_all.py` — feature ablation driver, gates, actual-result docs.
- `plots.py` — `figs/01..09`.
- `prepare_data.py` — cache builder + `^GSPC` inspection CSVs.
- `tests/` — feature/leakage/contract tests and fuzzy-controller tests.
- `archive/2026-10-21_pre_feature_update/` — historical pre-update results,
  fixed-LR logs, and figures. These are not results of the current model.

Run tests with:

```bash
python -m unittest discover -s tests
```

## Executed verification run (2026-10-21)

Command actually run:

```bash
python main.py --universe gspc --feature-sets paper full
```

This is a complete 30-seed run of the original `^GSPC` experiment with both
feature sets and fuzzy LR only. `results.csv` and `run_config.json` reflect
this run; the default full command above uses the pooled universe and has not
been run to completion in this environment.

| feature set | AUC | accuracy | balanced acc | precision | recall | F1 | MCC | majority acc |
|---|---|---|---|---|---|---|---|---|
| paper (9) | 0.5168 ± 0.0128 | 0.4793 ± 0.0370 | 0.5043 ± 0.0122 | 0.5829 ± 0.0304 | 0.3458 ± 0.2084 | 0.3971 ± 0.1803 | 0.0091 ± 0.0275 | 0.5791 |
| full (20) | 0.5103 ± 0.0188 | 0.4825 ± 0.0432 | 0.5056 ± 0.0135 | 0.6016 ± 0.0466 | 0.3596 ± 0.2810 | 0.3858 ± 0.2122 | 0.0158 ± 0.0369 | 0.5791 |

Test confusion matrices summed over 30 seeds:

- `paper`: `[[3281, 1669], [4455, 2355]]` (TN, FP / FN, TP).
- `full`: `[[3225, 1725], [4361, 2449]]`.

The paired `full - paper` test-AUC difference is `-0.00644` (95% CI
`[-0.01579, 0.00257]`, paired-t `p=0.1915`). This is a null/negative result:
there is no evidence that the added features improve test AUC, and both
feature sets are below the majority-class accuracy. The full-feature
validation log-loss is higher (worse), but the Holm-corrected secondary test
is not significant.

The pre-update archive contains an older 10-seed 7-feature fuzzy run
(`AUC 0.4941 ± 0.0123`). It is historical context only, not a like-for-like
baseline: the feature set, macro merge, and window row count differ. The
in-run `paper` set is the original-feature baseline for this task.
