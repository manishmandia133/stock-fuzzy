"""Project-wide constants. Torch is imported lazily so data/fuzzy scripts run without it."""
import random

import numpy as np

# ---- Reproducibility ----
SEEDS = [0, 1, 2, 3, 4]


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

WINDOW = 30
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 6
GRAD_CLIP = 1.0

# ---- Model (deliberately small for ~1.9k windows) ----
HIDDEN = 32
NUM_LAYERS = 1
DROPOUT = 0.0

# Chronological split fractions (70/15/15)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ---- Features (stationary only, no raw OHLC) ----
FEATURE_COLS = ["return_1d", "hl_range", "oc_ret", "vol_chg", "volatility_14d"]
TARGET_COL = "target_direction"
VOL_COL = "volatility_14d"
VOL_COL_IDX = FEATURE_COLS.index(VOL_COL)

# ---- Active learning-rate methods (5 fixed grid + 2 fuzzy) ----
# Five fixed-LR baselines. BEST_FIXED is chosen AFTER training using
# validation loss only (lowest mean best_val_loss across seeds), never test.
FIXED_LR = 1e-3
FIXED_LR_GRID = [2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3]
FIXED_LR_VARIANTS = {
    "fixed_2e-4": 2e-4,
    "fixed_5e-4": 5e-4,
    "fixed_1e-3": 1e-3,
    "fixed_1p5e-3": 1.5e-3,
    "fixed_2e-3": 2e-3,
}

# Trend input: validation-loss slope, normalised by 0.005 and clipped to [-1, 1].
TREND_SCALE = 0.005
TREND_MIN = -1.0
TREND_MAX = 1.0

# ---- Prior Sugeno prototype anchors (reference only; not trained) ----
LR_LOW = 1.5e-3
LR_MED = 1.0e-3
LR_HIGH = 0.4e-3

# ---- Output LR universe (Mamdani consequent) ----
LR_MIN = 0.0
LR_MAX = 2e-3
LR_SLOW_C = 0.4e-3
LR_MED_C = 1.0e-3
LR_FAST_C = 1.6e-3
