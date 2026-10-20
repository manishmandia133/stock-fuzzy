"""Project-wide constants. Torch is imported lazily so data/fuzzy scripts run without it."""
import random

import numpy as np

# ---- Reproducibility ----
SEEDS = list(range(10))


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
    """Resolve torch device lazily (CPU if torch/CUDA unavailable)."""
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


DEVICE = None  # resolved lazily via get_device(); kept for backward compat


def resolve_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = get_device()
    return DEVICE


# ---- Data scope: ^GSPC only ----
UNIVERSE = ["^GSPC"]
EXTENSION = False

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"  # fixed: report numbers are tied to this end date (yfinance end is exclusive)
VIX_TICKER = "^VIX"

WINDOW = 20
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 15
MIN_EPOCHS = 10
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4

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
TARGET_COL = "target_direction"
VOL_COL = "volatility_14d"

# ---- Active learning-rate methods ----
# fixed_1e-3: constant baseline (never changes).
# fuzzy: Mamdani controller below (low vol -> higher LR, high vol -> lower LR).
FIXED_LR = 1e-3

# ---- Output LR universe (Mamdani consequent) ----
LR_MIN = 1e-4
LR_MAX = 3e-3
LR_SLOW_C = 0.4e-3
# Requested Med/Fast centroids 1.2e-3/2.4e-3 put the surface mean at
# 1.403e-3 (>20% above 1e-3). These values keep the mean at 1.133e-3.
LR_MED_C = 0.8e-3
LR_FAST_C = 1.6e-3
