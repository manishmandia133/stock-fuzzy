"""Project-wide configuration. ^GSPC regression task only."""
import random
import numpy as np


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


_DEVICE = None


def resolve_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = get_device()
    return _DEVICE


# ── Data ───────────────────────────────────────────────────────────────────
TICKER      = "^GSPC"
VIX_TICKER  = "^VIX"
USDX_TICKER = "DX-Y.NYB"
FRED_SERIES = ["EFFR", "UNRATE", "UMCSENT", "T10Y2Y"]

START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"

# Chronological split: 80 % train+val | 20 % test
# Within train+val: last 20 % → validation
TRAIN_VAL_FRAC      = 0.80
VAL_FRAC_OF_TRAIN   = 0.20

# ── Features ───────────────────────────────────────────────────────────────
CORE_FEATURES = [
    "Close", "MACD", "RSI", "ATR", "VIX", "USDX", "EFFR", "UNRATE", "UMCSENT",
]
EXTRA_FEATURES = [
    "log_ret_1d", "log_ret_5d", "log_ret_20d",
    "prev_close", "vix_z20", "T10Y2Y",
    "vol_change", "close_ma50_ratio", "ma50_ma200_ratio",
    "realized_vol_20d",
]
ALL_FEATURES = CORE_FEATURES + EXTRA_FEATURES
FEATURE_COLS = ALL_FEATURES      # default

TARGET_COL = "log_return_next"   # log(Close_{t+1} / Close_t)
VOL_COL    = "realized_vol_20d"  # regime signal for fuzzy controller

CLIP = 3.0  # StandardScaler clip after train-only fit

# ── Model ──────────────────────────────────────────────────────────────────
WINDOW      = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
DROPOUT     = 0.2

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS        = 100
PATIENCE      = 20
MIN_EPOCHS    = 10
BATCH_SIZE    = 16
GRAD_CLIP     = 1.0
WEIGHT_DECAY  = 1e-4
EARLY_STOP_MA = 3
LOSS          = "huber"   # "mse" | "huber"
HUBER_DELTA   = 1.0

# ── Fuzzy LR universe (Mamdani consequent) ─────────────────────────────────
INITIAL_LR = 1e-3
LR_MIN     = 1e-4
LR_MAX     = 3e-3
LR_SLOW_C  = 0.4e-3
LR_MED_C   = 0.8e-3
LR_FAST_C  = 1.6e-3

# ── Active model ───────────────────────────────────────────────────────────
# Fuzzy_Input – membership degrees appended to every input timestep.
VARIANTS = ["Fuzzy_Input"]

# ── Seeds ──────────────────────────────────────────────────────────────────
SEEDS = list(range(10))   # default; override with --seeds N
