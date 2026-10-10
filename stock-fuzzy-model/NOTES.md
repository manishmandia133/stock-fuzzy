# NOTES.md — gate log (document all gate passes/failures here)

## T1 Reproducibility
- `config.set_seed(s)` seeds random/numpy (+ torch if installed). Torch is
  imported lazily, so `prepare_data.py` / fuzzy code run without it.
- Train windows are sorted by `end_date`, batch ORDER permuted per epoch.
- Gate: data build twice -> identical sums. PASS (sum=217426.6097).

## T2 Features
- Stationary: return_1d, hl_range, oc_ret, vol_chg, volatility_14d.
- Assertions: chronological split; 3-window label spot-check;
  class balance + majority baseline printed. PASS.

## T3 Scope / data (2026-10-10 run)
- `UNIVERSE = ["^GSPC"]`, 70/15/15 chronological split,
  `START=2015-01-01`, `END=2026-10-01` (yfinance end exclusive,
  `auto_adjust=True`).
- `data/raw_GSPC.csv`: shape (2953, 6), 0 NaNs (raw as returned, flattened).
- `data/processed_GSPC.csv`: shape (2936, 20), 0 NaNs; up-rates
  train 0.5319 / val 0.5682 / test 0.5442 (row level).
- Window level: train 1995 / val 380 / test 381 windows; up-rates
  0.5333 / 0.5711 / 0.5512. Gate n_train > 1500, no NaN: PASS.

## T4 Fuzzy controller (real Mamdani FIS)
- Input vol_rank: Low/Medium/High triangular Ruspini partition (sums to 1).
- Output LR universe [0, 2e-3]: Slow/Med/Fast triangles, centroids
  0.4e-3 / 1.0e-3 / 1.6e-3. Rules Low→Fast, Med→Med, High→Slow;
  min implication, max aggregation, centroid defuzz; 201-pt lookup table.
- Gate 2026-10-10: max_step=1.03e-05, range=1.2e-03, mean=1.000e-03,
  smooth/mono/mean PASS; Ruspini maxdiff=0.0 PASS;
  skfuzzy ControlSystem maxdiff≈3.1e-17 (< 1e-9) PASS. **Overall PASS.**
- NOTE (tautological): this gate and the corr(vol,lr) < -0.7 gate only verify
  the controller curve shape, NOT that fuzzy LRs help the LSTM.

## T5 Regime batching
- Contiguous batches, batch ORDER shuffled per epoch; final partial batch
  included (no drop_last). `BATCH_SIZE=32`, `EPOCHS=40`, `PATIENCE=6`.
- Gate: std of batch vol > 0.15. PASS (0.253 every run).

## T6 Training — 5 variants (2026-10-10)
- `fuzzy`, `fixed_1e-3`, `fixed_mean_fuzzy`, `fuzzy_inverted`,
  `fuzzy_shuffled_vol` (empirical train-vol → fuzzy LR, decoupled).
  Per-batch logs `logs/{variant}_s{seed}.csv` + histories
  `logs/{variant}_s{seed}_hist.csv`; best-val-loss weights; grad clip 1.0.
- Gate (fuzzy): corr(vol,lr) < -0.7. PASS (-0.996 every seed; inverted
  +0.996; fixed NaN as expected; shuffled ≈ 0). Tautological — see T4 note.
- **Honest outcome: val_loss ≈ ln2 (0.684–0.687) on all 25 runs; every run
  is a constant predictor** (pred_pos_rate=1.0, pred_std≈0.001,
  test_acc == majority 0.551, MCC=0.0). Reported as-is, not as a win.

## T7 Evaluation
- `train_val_gap = val_loss - train_loss` (≈ -0.0067; train slightly above
  val under dropout). AUC > 0.60 → `leakage_flag`: none triggered.
  LogReg-on-last-day AUC ≈ 0.50 expected (see results.csv `logreg_auc`).
- Bootstrap (2000 resamples) 95% CI per seed in `test_auc_lo/hi`; all CIs
  cover 0.5 except fuzzy_shuffled_vol s4 (AUC 0.564 [0.507, 0.622]) —
  a single-seed outlier, still a constant predictor at threshold 0.5.

## T8 Stats (5 seeds; p-values UNCORRECTED)
- Mean±std test AUC: fuzzy 0.488±0.024; fixed_1e-3 0.488±0.024;
  fixed_mean_fuzzy 0.488±0.024; fuzzy_inverted 0.488±0.025;
  fuzzy_shuffled_vol 0.508±0.038.
- fuzzy vs fixed_1e-3 (= vs fixed_mean_fuzzy): test_auc t p=0.7251 /
  W p=0.6250; best_val_loss t p=0.3417 / W p=0.4375;
  highvol_loss_var t p=0.0002 / W p=0.0625 (Wilcoxon floor at n=5).
- fuzzy vs fuzzy_shuffled_vol: test_auc t p=0.2799 / W p=0.3125;
  best_val_loss t p=0.6406 / W p=0.8125; highvol_loss_var t p=0.1690 /
  W p=0.1875.
- Decision: "fuzzy ≈ fixed on AUC" → honest null; high-vol loss-variance
  t-test hint needs more seeds before any stability claim.

## Run log — ^GSPC 2015-01-01..2026-10-01, seeds 0-4 x 5 variants (2026-10-10)
- Gates: T1 PASS, T2 PASS, T3 PASS, T4 PASS, T5 PASS, T6 PASS (with
  tautology note), T7 no leakage flags. **No gate failed.**
- Cleanup done: stale `logs/`, `results.csv`, `figs/`, `legacy/`,
  `__pycache__` deleted; `cosine`/`crisp` variants removed so code, results,
  README and NOTES are consistent (5 variants everywhere).
- Figs: `figs/01_membership_input.png` … `07_auc_boxplot.png`
  (+ `08_lr_vs_vol_s0.png`, `09_lr_over_step_s0.png`).
- `plot_val_curves` (06) shows train AND val loss per variant (mean±std)
  with ln2 line; `lr_over_step` x-axis labelled as shuffled optimizer steps;
  boxplot uses `tick_labels` with `labels` fallback for matplotlib<3.9.

## File map
- New pipeline: `config.py`, `prepare_data.py`, `data.py`, `fuzzy.py`,
  `model.py`, `train.py`, `evaluate.py`, `run_all.py`, `plots.py`.
  `main.py` is a thin runner over `run_all.py`. No `legacy/` directory.
