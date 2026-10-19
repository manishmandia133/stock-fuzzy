"""Fuzzy LR controller: real Mamdani FIS (vol_rank -> LR).

Input vol_rank in [0, 1]: Low/Medium/High triangular Ruspini partition
  (memberships sum to 1 everywhere).
Output LR universe [0, 2e-3]: Slow/Medium/Fast symmetric triangles with
  centroids 0.4e-3 / 1.0e-3 / 1.6e-3.
Rules: Low -> Fast, Medium -> Medium, High -> Slow.
Inference: min implication, max aggregation, centroid defuzzification.
A 201-pt lookup table is precomputed for training; it is cross-checked
against skfuzzy.control.ControlSystem (max diff < 1e-9).
"""

import numpy as np

import config as cfg

V_GRID = np.linspace(0, 1, 201)
OUT_GRID = np.linspace(cfg.LR_MIN, cfg.LR_MAX, 2001)


def _trimf(x, a, b, c):
    """Triangular membership (a <= b <= c). Handles a==b / b==c shoulders."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # rising edge
    if b != a:
        rise = (x - a) / (b - a)
        m = (x >= a) & (x <= b)
        y[m] = np.maximum(y[m], np.clip(rise[m], 0, 1))
    else:
        y[x <= b] = np.maximum(y[x <= b], 1.0 * (x[x <= b] <= b))
    # falling edge
    if c != b:
        fall = (c - x) / (c - b)
        m = (x >= b) & (x <= c)
        y[m] = np.maximum(y[m], np.clip(fall[m], 0, 1))
    else:
        y[x >= b] = np.maximum(y[x >= b], 1.0 * (x[x >= b] >= b))
    return np.clip(y, 0, 1)


# ---- Input MFs: Ruspini triangular partition of [0, 1] ----
def mu_low(v):
    return _trimf(v, 0.0, 0.0, 0.5)


def mu_med(v):
    return _trimf(v, 0.0, 0.5, 1.0)


def mu_high(v):
    return _trimf(v, 0.5, 1.0, 1.0)


# ---- Output MFs: symmetric triangles, centroids at spec values ----
SLOW_MF = _trimf(OUT_GRID, 0.0, cfg.LR_SLOW_C, 0.8e-3)
MED_MF = _trimf(OUT_GRID, 0.6e-3, cfg.LR_MED_C, 1.4e-3)
FAST_MF = _trimf(OUT_GRID, 1.2e-3, cfg.LR_FAST_C, 2.0e-3)


def mamdani_lr(v) -> float:
    """Single-point Mamdani inference with centroid defuzzification."""
    v = float(np.clip(v, 0, 1))
    a_low, a_med, a_high = mu_low(v), mu_med(v), mu_high(v)
    # Rules: Low -> Fast, Medium -> Medium, High -> Slow (min implication)
    imp_fast = np.minimum(a_low, FAST_MF)
    imp_med = np.minimum(a_med, MED_MF)
    imp_slow = np.minimum(a_high, SLOW_MF)
    agg = np.maximum(np.maximum(imp_fast, imp_med), imp_slow)  # max aggregation
    denom = agg.sum()
    if denom <= 0:
        return float((cfg.LR_SLOW_C + cfg.LR_MED_C + cfg.LR_FAST_C) / 3.0)
    return float((agg * OUT_GRID).sum() / denom)  # centroid defuzz


# Precomputed 201-pt lookup table for training
LR_CURVE = np.array([mamdani_lr(v) for v in V_GRID])
MEAN_LR = float(LR_CURVE.mean())


def fuzzy_lr(v) -> float:
    """Map vol_rank in [0,1] -> LR via linear interp on the precomputed curve."""
    return float(np.interp(np.clip(v, 0, 1), V_GRID, LR_CURVE))


def skfuzzy_reference_curve():
    """Build the identical FIS with skfuzzy.control and evaluate on V_GRID."""
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    vol = ctrl.Antecedent(V_GRID, "vol")
    lr = ctrl.Consequent(OUT_GRID, "lr")
    vol["Low"] = _trimf(V_GRID, 0.0, 0.0, 0.5)
    vol["Medium"] = _trimf(V_GRID, 0.0, 0.5, 1.0)
    vol["High"] = _trimf(V_GRID, 0.5, 1.0, 1.0)
    lr["Slow"] = _trimf(OUT_GRID, 0.0, cfg.LR_SLOW_C, 0.8e-3)
    lr["Medium"] = _trimf(OUT_GRID, 0.6e-3, cfg.LR_MED_C, 1.4e-3)
    lr["Fast"] = _trimf(OUT_GRID, 1.2e-3, cfg.LR_FAST_C, 2.0e-3)
    r1 = ctrl.Rule(vol["Low"], lr["Fast"])
    r2 = ctrl.Rule(vol["Medium"], lr["Medium"])
    r3 = ctrl.Rule(vol["High"], lr["Slow"])
    sys = ctrl.ControlSystem([r1, r2, r3])
    sim = ctrl.ControlSystemSimulation(sys)
    out = np.zeros_like(V_GRID)
    for i, v in enumerate(V_GRID):
        sim.input["vol"] = float(v)
        sim.compute()
        out[i] = float(sim.output["lr"])
    _ = fuzz  # keep import used (membership math mirrors skfuzzy.trimf)
    return out


def check_gate() -> dict:
    """Gate: smooth, monotone decreasing, mean within 20% of 1e-3.

    NOTE: these gates are tautological w.r.t. training benefit — they only
    verify the controller curve shape, not that fuzzy LRs help the LSTM.
    """
    diffs = np.abs(np.diff(LR_CURVE))
    max_step = float(diffs.max())
    lr_range = float(LR_CURVE.max() - LR_CURVE.min())
    mean_ok = abs(MEAN_LR - 1e-3) / 1e-3 < 0.20
    smooth_ok = max_step < 0.05 * lr_range
    mono_ok = bool(np.all(np.diff(LR_CURVE) <= 1e-12))
    # Ruspini partition check
    part = mu_low(V_GRID) + mu_med(V_GRID) + mu_high(V_GRID)
    ruspini_maxdiff = float(np.abs(part - 1.0).max())
    ruspini_ok = ruspini_maxdiff < 1e-9
    # Cross-check against skfuzzy ControlSystem
    try:
        ref = skfuzzy_reference_curve()
        sk_maxdiff = float(np.abs(ref - LR_CURVE).max())
        sk_ok = sk_maxdiff < 1e-9
    except Exception as e:  # noqa: BLE001
        sk_maxdiff = float("nan")
        sk_ok = False
        print(f"skfuzzy cross-check error: {e}")
    ok = smooth_ok and mean_ok and mono_ok and ruspini_ok and sk_ok
    return {"max_step": max_step, "range": lr_range, "mean": MEAN_LR,
            "smooth_ok": smooth_ok, "mean_ok": mean_ok, "mono_ok": mono_ok,
            "ruspini_maxdiff": ruspini_maxdiff, "ruspini_ok": ruspini_ok,
            "skfuzzy_maxdiff": sk_maxdiff, "skfuzzy_ok": sk_ok,
            "pass": bool(ok)}


# ---------------------------------------------------------------------------
# ORIGINAL zero-order Sugeno controller (prior prototype, kept as reference).
# Membership-weighted average of 3 constants — NOT used for training anymore.
# MFs: trapmf[0,0,.25,.55], trimf[.2,.5,.8], trapmf[.45,.75,1,1];
# anchors: low vol -> 1.5e-3, med -> 1.0e-3, high -> 0.4e-3.
# ---------------------------------------------------------------------------

def _trapmf(x, a, b, c, d):
    """Trapezoidal membership (plateau on [b, c]; shoulders if a==b / c==d)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    plat = (x >= b) & (x <= c)
    y[plat] = 1.0
    if d != c:
        m = (x > c) & (x < d)
        y[m] = (d - x[m]) / (d - c)
    return np.clip(y, 0, 1)


SUGENO_LOW = _trapmf(V_GRID, 0, 0, 0.25, 0.55)
SUGENO_MED = _trimf(V_GRID, 0.20, 0.50, 0.80)
SUGENO_HIGH = _trapmf(V_GRID, 0.45, 0.75, 1.0, 1.0)
SUGENO_CURVE = ((SUGENO_LOW * cfg.LR_LOW + SUGENO_MED * cfg.LR_MED
                 + SUGENO_HIGH * cfg.LR_HIGH)
                / (SUGENO_LOW + SUGENO_MED + SUGENO_HIGH + 1e-9))
SUGENO_MEAN = float(SUGENO_CURVE.mean())


def sugeno_lr(v) -> float:
    """Reference: original Sugeno weighted-average mapping (not for training)."""
    return float(np.interp(np.clip(v, 0, 1), V_GRID, SUGENO_CURVE))


def sugeno_gate() -> dict:
    """Gate numbers of the ORIGINAL Sugeno curve (report traceability)."""
    diffs = np.abs(np.diff(SUGENO_CURVE))
    max_step = float(diffs.max())
    lr_range = float(SUGENO_CURVE.max() - SUGENO_CURVE.min())
    mean_ok = abs(SUGENO_MEAN - 1e-3) / 1e-3 < 0.20
    smooth_ok = max_step < 0.05 * lr_range
    mono_ok = bool(np.all(np.diff(SUGENO_CURVE) <= 1e-12))
    return {"max_step": max_step, "range": lr_range, "mean": SUGENO_MEAN,
            "smooth_ok": smooth_ok, "mean_ok": mean_ok, "mono_ok": mono_ok,
            "pass": bool(smooth_ok and mean_ok and mono_ok)}
