"""Fuzzy LR controller as precomputed lookup table (PLAN T4).

No per-batch scikit-fuzzy inference, no try/except fallback.
High volatility -> lower LR.
"""
import numpy as np
import skfuzzy as fuzz

import config as cfg

V_GRID = np.linspace(0, 1, 201)
_low = fuzz.trapmf(V_GRID, [0, 0, 0.25, 0.55])
_med = fuzz.trimf(V_GRID, [0.20, 0.50, 0.80])
_high = fuzz.trapmf(V_GRID, [0.45, 0.75, 1, 1])

LR_LOW, LR_MED, LR_HIGH = cfg.LR_LOW, cfg.LR_MED, cfg.LR_HIGH
LR_CURVE = (_low * LR_LOW + _med * LR_MED + _high * LR_HIGH) / (_low + _med + _high + 1e-9)

MEAN_LR = float(LR_CURVE.mean())


def fuzzy_lr(v, invert=False) -> float:
    """Map vol_rank in [0,1] -> LR via linear interp on the precomputed curve."""
    c = LR_CURVE[::-1] if invert else LR_CURVE
    return float(np.interp(np.clip(v, 0, 1), V_GRID, c))


def check_gate() -> dict:
    """T4 gate: smooth, mean within 20% of 1e-3, monotonically decreasing."""
    diffs = np.abs(np.diff(LR_CURVE))
    max_step = float(diffs.max())
    lr_range = float(LR_CURVE.max() - LR_CURVE.min())
    mean_ok = abs(MEAN_LR - 1e-3) / 1e-3 < 0.20
    smooth_ok = max_step < 0.05 * lr_range
    mono_ok = bool(np.all(np.diff(LR_CURVE) <= 1e-12))
    return {"max_step": max_step, "range": lr_range, "mean": MEAN_LR,
            "smooth_ok": smooth_ok, "mean_ok": mean_ok, "mono_ok": mono_ok,
            "pass": smooth_ok and mean_ok and mono_ok}
