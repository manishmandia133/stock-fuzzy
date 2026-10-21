"""Reference targets from Bhandari et al. 2022.

Hardcoded only here. No model code. Persistence is computed from data.
Paper window: ^GSPC 2006-01-03 .. 2021-09-29, chronological 80/20 train/test.
Our window:   ^GSPC 2015-01-01 .. 2025-12-31.
Comparison is aspirational; always compute persistence from actual data.
"""

# --- Paper (Bhandari et al. 2022), best single-layer LSTM ---
PAPER_RMSE      = 40.4574
PAPER_RMSE_BEST = 37.9416   # best seed
PAPER_MSE       = 1637.0
PAPER_MAPE      = 0.7989    # percent
PAPER_R         = 0.9976
PAPER_MAE       = None      # not reported

# --- Approximate persistence on the paper's window (informational) ---
APPROX_PERSIST_RMSE = 41.5
APPROX_PERSIST_MAPE = 0.86   # percent
APPROX_PERSIST_R    = 0.9973

NOTE = (
    "Paper numbers come from Bhandari et al. (2022), Table III. "
    "Our test window (2015-2025) is different; beat both paper AND "
    "the persistence computed from actual data."
)
