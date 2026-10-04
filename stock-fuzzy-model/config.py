"""Project-wide constants (PLAN T1 + T3)."""
import random

import numpy as np
import torch

# ---- Reproducibility ----
SEEDS = [0, 1, 2, 3, 4]


def set_seed(s: int) -> None:
    """Seed numpy, torch (+cuda) and stdlib random. No pre-shuffle here."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data scope (T3: ^GSPC first; 99-ticker pool only via EXTENSION) ----
UNIVERSE = ["^GSPC"]
EXTENSION = False  # set True only for T10

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"  # fixed, never None (T1 gate)

WINDOW = 60
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 5
GRAD_CLIP = 1.0

# Chronological split fractions for ^GSPC (70/15/15)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ---- Features (T2: stationary only, no raw OHLC) ----
FEATURE_COLS = ["return_1d", "hl_range", "oc_ret", "vol_chg", "volatility_14d"]
TARGET_COL = "target_direction"
VOL_COL = "volatility_14d"
VOL_COL_IDX = FEATURE_COLS.index(VOL_COL)

# ---- LR constants (T4) ----
LR_LOW = 1.5e-3   # low vol  -> higher LR
LR_MED = 1.0e-3   # med vol  -> medium LR
LR_HIGH = 0.4e-3  # high vol -> lower LR
FIXED_LR = 1e-3
