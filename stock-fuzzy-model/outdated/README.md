# Outdated artifacts — do not use as current results

Two archived experiments live here. Neither is part of the active experiment.

## `outdated/` (top level): seven-variant phase
Files under `outdated/figs/` and `outdated/logs/` come from the previous
seven-variant experiment (`fuzzy`, `fixed_1e-3`, `fixed_mean_fuzzy`, `cosine`,
`crisp`, `fuzzy_inverted`, `fuzzy_shuffled_vol`). They are retained only for
history. Do not quote them as results of the current experiment.

## `outdated/two_method/`: two-method phase
Contains the later two-method experiment (`fixed_1e-3` vs `fuzzy`), including
its `results.csv`, `run_config.json`, `run_output.log`, `logs/`, and `figs/`.
That experiment produced null results: every run was a constant predictor.
It is superseded by the current 5 fixed-LR grid + 2 fuzzy variants design.

## Active experiment
The active code trains seven variants:

1. `fixed_2e-4`
2. `fixed_5e-4`
3. `fixed_1e-3`
4. `fixed_1p5e-3`
5. `fixed_2e-3`
6. `fuzzy` — two-input Mamdani (`vol_rank`, validation-loss trend)
7. `fuzzy_vol_only` — one-input ablation using only `vol_rank`

Current outputs are written to `results.csv`, `run_config.json`, `logs/`, and
`figs/` by `python main.py`. `BEST_FIXED` is selected from the fixed grid using
validation loss only, never test metrics.
