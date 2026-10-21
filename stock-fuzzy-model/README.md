# Stock Fuzzy-LSTM Model

Volatility-aware fuzzy learning-rate control for an LSTM predicting a
next-day **volatility** label by default. Soft Computing lab.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # cache universe + ^GSPC inspection CSV
python main.py           # 30 seeds x 5 LR variants -> results.csv + figs/
```

`main.py` writes `results.csv`, `run_config.json`, `NOTES.md`, and
`REPORT_UPDATES.md` from the actual run. Numbers are never hand-entered.

## Targets (`config.TARGET_MODE`)

- `"vol"` (default): label `1` if next-day `hl_range` is above the
  **TRAIN-set median**, else `0`. The median is computed on train only and
  the label stays aligned with window end `t`.
- `"direction"`: label `1` if `Close[t+1] > Close[t]`, else `0`.

## Universes (`config.UNIVERSE_MODE`)

- `"pooled"` (default): the 100-company source list (60 US + 40 India;
  99 usable symbols — `TATAMOTORS.NS` is currently delisted). Windows are
  built per ticker, then
  concatenated. **One global set of chronological cut dates** is used for
  every ticker. The scaler, vol_rank reference, and vol-target median are
  fit on TRAIN only.
- `"gspc"`: the original `^GSPC`-only experiment.

## LR variants

| variant | rule |
|---|---|
| `fixed_3e-4` | constant 3e-4 |
| `fixed_1e-3` | constant 1e-3 |
| `fixed_3e-3` | constant 3e-3 |
| `fuzzy_dyn` | per-epoch Mamdani controller on relative val-loss change + epoch mean vol_rank; 3-epoch linear warmup |
| `fuzzy_vol` | per-batch Mamdani `vol_rank -> LR` curve (ablation) |

No ReduceLROnPlateau, cosine decay, or other scheduler is used.

## Training / evaluation protocol

- Chronological 70/15/15 split; `StandardScaler` fit on TRAIN only; clip
  to `[-3, 3]`.
- Same seed -> same initialisation and same batch order for every variant
  (paired design).
- Early stopping on the **3-epoch moving-average validation loss** (not
  validation AUC). Best weights are restored only from
  `best_epoch >= MIN_EPOCHS (10)`.
- `fuzzy_dyn`: `Improving/Worse/Flat` from
  `dloss=(vloss-prev_vloss)/prev_vloss` and `LowVol/HighVol` from vol_rank;
  3x2 rules with min-AND and weighted-average defuzzification; LR multiplier
  in `[0.40, 1.15]`; `lr = clip(lr*factor, LR_MIN, LR_MAX)`.
- Classification threshold is selected on VALIDATION only
  (balanced accuracy over `0.40..0.60`).
- **Primary metric (fixed in advance): test AUC.** Secondary metrics:
  test log-loss, epochs-to-target val loss, area under the val-loss curve,
  high-vol test AUC, and across-seed std. Only the primary decides the
  claim. Secondary paired tests get Holm correction; the primary is
  uncorrected.
- Claim rule: `fuzzy_dyn` beats the validation-selected best fixed LR only
  if primary paired-t `p < 0.05`, the 95% bootstrap CI excludes 0, and the
  direction favours fuzzy. Otherwise the run prints `NULL RESULT`.

## Files

- `config.py` — modes, universe, dates, features, targets, LR constants,
  fuzzy rule table.
- `data.py` — cached downloads, features, train-only scaling/median/vol_rank,
  pooled global date cuts, window build sorted by end date, T1-T3 asserts.
- `fuzzy.py` — Mamdani `fuzzy_vol` FIS + `fuzzy_dyn` per-epoch controller.
- `train.py` — paired regime batching, early stopping, LR application and
  integrity checks.
- `evaluate.py` — primary/secondary metrics, bootstrap CI, paired tests,
  Holm correction.
- `run_all.py` — 30-seed x 5-variant driver, validation-only best-fixed
  selection, claim rule, plots, auto-generated docs.
- `plots.py` — `figs/01..09`.
- `tests/test_fuzzy.py` — `python -m unittest discover -s tests`.
