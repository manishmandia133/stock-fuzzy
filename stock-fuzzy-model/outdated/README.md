# Outdated artifacts — do not use as current results

These files come from the **previous seven-variant experiment**
(`fuzzy`, `fixed_1e-3`, `fixed_mean_fuzzy`, `cosine`, `crisp`,
`fuzzy_inverted`, `fuzzy_shuffled_vol`).

The active experiment now trains only:

1. `fixed_1e-3` — constant learning rate `0.001`
2. `fuzzy` — Mamdani fuzzy learning rate driven by batch volatility

`results_7variants.csv`, `logs/`, and `figs/` were generated before the
removal of the five deprecated methods. They are retained only for history.
They MUST NOT be quoted or compared as results of the current experiment.
Current outputs are written to `results.csv`, `run_config.json`, `logs/`,
and `figs/` by `python main.py`.
