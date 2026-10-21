"""Project-wide constants. Torch is imported lazily so data/fuzzy scripts run without it."""
import random

import numpy as np

# ---- Reproducibility ----
SEEDS = list(range(30))


def set_seed(s: int) -> None:
    """Seed stdlib random + numpy, and torch if available."""
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
    """Resolve torch device lazily (CPU if torch/CUDA/MPS unavailable)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


DEVICE = None  # resolved lazily via get_device(); kept for backward compat


def resolve_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = get_device()
    return DEVICE


# ---- Data scope -----------------------------------------------------------
# "pooled" = the notebook's 100-company source list (60 US + 40 India).
# TATAMOTORS.NS is currently delisted, leaving the intended 99 usable tickers;
# the actual count is printed at runtime. "gspc" keeps the original
# ^GSPC-only experiment available.
UNIVERSE_MODE = "pooled"  # "pooled" | "gspc"
UNIVERSE = ["^GSPC"]  # used only when UNIVERSE_MODE == "gspc"

US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "VZ", "ADBE",
    "CMCSA", "NFLX", "KO", "NKE", "MRK", "PFE", "T", "INTC", "CSCO", "XOM",
    "CVX", "WMT", "MCD", "ABT", "CRM", "ORCL", "IBM", "QCOM", "TXN", "HON",
    "AMD", "COST", "PEP", "ABBV", "TMO", "AVGO", "ACN", "LLY", "DHR", "NEE",
    "LIN", "UNP", "PM", "RTX", "LOW", "UPS", "SBUX", "BA", "GE", "CAT",
]

INDIA_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS",
    "GRASIM.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BRITANNIA.NS",
]

POOLED_TICKERS = US_TICKERS + INDIA_TICKERS

EXTENSION = UNIVERSE_MODE == "pooled"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"  # fixed: report numbers are tied to this end date (yfinance end is exclusive)
VIX_TICKER = "^VIX"

WINDOW = 20
# Pooled mode follows the 100-company notebook batch size; ^GSPC keeps 32.
BATCH_SIZE = 128 if UNIVERSE_MODE == "pooled" else 32
EPOCHS = 80
PATIENCE = 15
MIN_EPOCHS = 10
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
EARLY_STOP_MA = 3  # moving-average window for validation-loss early stopping
WARMUP_EPOCHS = 3  # fuzzy_dyn linear warmup (epochs)

# Standardised features are clipped to [-CLIP, CLIP] after train-only fit.
CLIP = 3.0

# ---- Model ----
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.3

# ---- Validation threshold selection (no test tuning) ----
THRESHOLD_LOW = 0.40
THRESHOLD_HIGH = 0.60
THRESHOLD_STEP = 0.005

# Chronological split fractions (70/15/15)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ---- Features (stationary + VIX; no raw OHLC) ----
FEATURE_COLS = ["return_1d", "hl_range", "oc_ret", "vol_chg", "volatility_14d",
                "vix_level", "vix_z20"]
VOL_COL = "volatility_14d"

# ---- Target ---------------------------------------------------------------
# "direction": Close[t+1] > Close[t].
# "vol":       1 if next-day hl_range > TRAIN-set median, else 0.
TARGET_MODE = "vol"  # "vol" | "direction"
VOL_TARGET_SOURCE = "hl_range"  # "hl_range" or "abs_return"
DIRECTION_TARGET_COL = "target_direction"
VOL_TARGET_COL = "target_vol"
VOL_TARGET_RAW_COL = "target_vol_raw"  # next-day hl_range / |return| (no median applied)
TARGET_COL = VOL_TARGET_COL if TARGET_MODE == "vol" else DIRECTION_TARGET_COL

# ---- Active learning-rate methods ----------------------------------------
# Three fixed baselines (the validation-selected best is used for the paired
# comparison) plus two fuzzy ablations. No other scheduler is used.
FIXED_LRS = {"fixed_3e-4": 3e-4, "fixed_1e-3": 1e-3, "fixed_3e-3": 3e-3}
FIXED_LR = 1e-3  # base / default fixed LR and fuzzy_dyn starting LR

VARIANTS = ["fixed_3e-4", "fixed_1e-3", "fixed_3e-3", "fuzzy_dyn", "fuzzy_vol"]
FUZZY_VARIANTS = ["fuzzy_dyn", "fuzzy_vol"]

# ---- Output LR universe (Mamdani consequent, fuzzy_vol) ----
LR_MIN = 1e-4
LR_MAX = 3e-3
LR_SLOW_C = 0.4e-3
# Requested Med/Fast centroids 1.2e-3/2.4e-3 put the surface mean at
# 1.403e-3 (>20% above 1e-3). These values keep the mean at 1.133e-3.
LR_MED_C = 0.8e-3
LR_FAST_C = 1.6e-3

# ---- fuzzy_dyn controller -------------------------------------------------
FUZZY_DYN_DLOSS_SCALE = 0.005
FUZZY_DYN_RULES = {
    ("improving", "low"): 1.15,
    ("improving", "high"): 1.00,
    ("flat", "low"): 0.90,
    ("flat", "high"): 0.70,
    ("worse", "low"): 0.60,
    ("worse", "high"): 0.40,
}
