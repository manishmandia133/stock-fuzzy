"""Fuzzy logic components.

M2 – vol_rank → LR (Mamdani single-input FIS, re-centred for Adam 1e-3).
"""
import numpy as np

import config as cfg

# ── Shared output universe ──────────────────────────────────────────────────
V_GRID   = np.linspace(0, 1, 201)
OUT_GRID = np.linspace(cfg.LR_MIN, cfg.LR_MAX, 2001)


# ── Triangular MF helper ────────────────────────────────────────────────────

def _trimf(x, a, b, c):
    """Triangular MF trimf(x; a, b, c). Handles shoulder (a==b or b==c)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        m = (x >= a) & (x <= b)
        y[m] = np.clip((x[m] - a) / (b - a), 0, 1)
    else:
        y[x <= b] = 1.0
    if c != b:
        m = (x >= b) & (x <= c)
        y[m] = np.maximum(y[m], np.clip((c - x[m]) / (c - b), 0, 1))
    else:
        y[x >= b] = np.maximum(y[x >= b], 1.0)
    return np.clip(y, 0, 1)


# ── Output MFs ──────────────────────────────────────────────────────────────
SLOW_MF = _trimf(OUT_GRID, 0.0,          cfg.LR_SLOW_C, cfg.LR_MED_C)
MED_MF  = _trimf(OUT_GRID, cfg.LR_SLOW_C, cfg.LR_MED_C,  cfg.LR_FAST_C)
FAST_MF = _trimf(OUT_GRID, cfg.LR_MED_C,  cfg.LR_FAST_C, cfg.LR_MAX)

_OUT_MFS = {"Slow": SLOW_MF, "Med": MED_MF, "Fast": FAST_MF}


# ── Centroid defuzzification ────────────────────────────────────────────────

def _centroid_exact(x, mf):
    """Exact centroid of a piecewise-linear MF."""
    moment, area = 0.0, 0.0
    for i in range(1, len(x)):
        x1, x2, y1, y2 = x[i - 1], x[i], mf[i - 1], mf[i]
        if y1 == y2 == 0.0 or x1 == x2:
            continue
        dx = x2 - x1
        if y1 == y2:
            moment += 0.5 * (x1 + x2) * dx * y1
            area   += dx * y1
        elif y1 == 0.0:
            moment += (2.0 / 3.0 * dx + x1) * 0.5 * dx * y2
            area   += 0.5 * dx * y2
        elif y2 == 0.0:
            moment += (1.0 / 3.0 * dx + x1) * 0.5 * dx * y1
            area   += 0.5 * dx * y1
        else:
            moment += (2.0 / 3.0 * dx * (y2 + 0.5 * y1) / (y1 + y2) + x1) \
                      * 0.5 * dx * (y1 + y2)
            area   += 0.5 * dx * (y1 + y2)
    return float(moment / area) if area > 0 else float("nan")


def _aggregate_and_defuzz(cuts_labels):
    """Apply min-implication / max-aggregation / centroid-defuzz.

    cuts_labels: list of (strength, label) where label ∈ {"Slow","Med","Fast"}.
    Returns LR (float).
    """
    agg = np.zeros_like(OUT_GRID)
    for strength, label in cuts_labels:
        if strength > 0:
            agg = np.maximum(agg, np.minimum(strength, _OUT_MFS[label]))
    if agg.sum() <= 0:
        return float((cfg.LR_SLOW_C + cfg.LR_MED_C + cfg.LR_FAST_C) / 3.0)
    return _centroid_exact(OUT_GRID, agg)


# ═══════════════════════════════════════════════════════════════════════════
# M2 – Single-input FIS: vol_rank → LR
# Rule: Low → Fast, Medium → Medium, High → Slow
# ═══════════════════════════════════════════════════════════════════════════

# Input MFs: Ruspini partition of [0,1]
def mu_low(v):  return _trimf(np.asarray(v, float), 0.0, 0.0, 0.5)
def mu_med(v):  return _trimf(np.asarray(v, float), 0.0, 0.5, 1.0)
def mu_high(v): return _trimf(np.asarray(v, float), 0.5, 1.0, 1.0)


def mamdani_lr(v) -> float:
    """Single-point M2 inference: vol_rank → LR."""
    v = float(np.clip(v, 0, 1))
    a_low, a_med, a_high = float(mu_low(v)), float(mu_med(v)), float(mu_high(v))
    return _aggregate_and_defuzz([
        (a_high, "Slow"),
        (a_med,  "Med"),
        (a_low,  "Fast"),
    ])


# Precomputed 201-pt lookup table
LR_CURVE = np.array([mamdani_lr(v) for v in V_GRID])
MEAN_LR  = float(LR_CURVE.mean())


def fuzzy_lr(v) -> float:
    """M2: vol_rank → LR via linear interp on precomputed curve."""
    return float(np.interp(np.clip(v, 0, 1), V_GRID, LR_CURVE))


# ═══════════════════════════════════════════════════════════════════════════
# Gate checks (T4) – curve shape only; mean within 20 % of 1e-3
# ═══════════════════════════════════════════════════════════════════════════

def check_gate() -> dict:
    """Verify M2 curve: smooth, monotone decreasing, mean ≈ 1e-3."""
    diffs    = np.diff(LR_CURVE)
    lr_range = float(LR_CURVE.max() - LR_CURVE.min())
    mean_ok  = abs(MEAN_LR - 1e-3) / 1e-3 < 0.20
    smooth   = float(np.abs(diffs).max()) < 0.05 * lr_range
    mono     = bool(np.all(diffs <= 1e-12))
    part     = mu_low(V_GRID) + mu_med(V_GRID) + mu_high(V_GRID)
    ruspini  = float(np.abs(part - 1.0).max()) < 1e-9
    ok = mean_ok and smooth and mono and ruspini
    return {
        "mean_lr":    MEAN_LR,
        "mean_ok":    mean_ok,
        "smooth_ok":  smooth,
        "mono_ok":    mono,
        "ruspini_ok": ruspini,
        "pass":       bool(ok),
    }
