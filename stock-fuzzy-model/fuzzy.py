"""Fuzzy LR controller: real Mamdani FIS (vol_rank -> LR).

Input vol_rank in [0, 1]: Low/Medium/High triangular Ruspini partition
  (memberships sum to 1 everywhere).
Output LR universe [1e-4, 3e-3]: overlapping Slow/Medium/Fast triangles.
  Centroids: 0.4e-3 / 0.8e-3 / 1.6e-3. The requested 1.2e-3 / 2.4e-3
  Med/Fast centroids produced a surface mean of 1.403e-3, outside the
  20% band around 1e-3, so Med/Fast were scaled down (surface mean
  1.133e-3).
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


def _interp_universe(x, mf, y):
    """Universe points where a piecewise-linear mf equals level y."""
    if y == 0.0:
        idx = np.where(np.diff(mf > y))[0]
    else:
        idx = np.where(np.diff(mf >= y))[0]
    return (x[idx] + (y - mf[idx]) * (x[idx + 1] - x[idx])
            / (mf[idx + 1] - mf[idx]))


def _centroid_exact(x, mf):
    """Exact centroid of a piecewise-linear MF (same math as skfuzzy)."""
    moment_sum, area_sum = 0.0, 0.0
    for i in range(1, len(x)):
        x1, x2 = x[i - 1], x[i]
        y1, y2 = mf[i - 1], mf[i]
        if y1 == y2 == 0.0 or x1 == x2:
            continue
        if y1 == y2:
            moment = 0.5 * (x1 + x2)
            area = (x2 - x1) * y1
        elif y1 == 0.0:
            moment = 2.0 / 3.0 * (x2 - x1) + x1
            area = 0.5 * (x2 - x1) * y2
        elif y2 == 0.0:
            moment = 1.0 / 3.0 * (x2 - x1) + x1
            area = 0.5 * (x2 - x1) * y1
        else:
            moment = ((2.0 / 3.0 * (x2 - x1) * (y2 + 0.5 * y1))
                      / (y1 + y2) + x1)
            area = 0.5 * (x2 - x1) * (y1 + y2)
        moment_sum += moment * area
        area_sum += area
    if area_sum <= 0.0:
        return float("nan")
    return float(moment_sum / area_sum)


# ---- Input MFs: Ruspini triangular partition of [0, 1] ----
def mu_low(v):
    return _trimf(v, 0.0, 0.0, 0.5)


def mu_med(v):
    return _trimf(v, 0.0, 0.5, 1.0)


def mu_high(v):
    return _trimf(v, 0.5, 1.0, 1.0)


# ---- Output MFs: overlapping triangles, centroids at the adjusted values ----
SLOW_MF = _trimf(OUT_GRID, 0.0, cfg.LR_SLOW_C, cfg.LR_MED_C)
MED_MF = _trimf(OUT_GRID, cfg.LR_SLOW_C, cfg.LR_MED_C, cfg.LR_FAST_C)
FAST_MF = _trimf(OUT_GRID, cfg.LR_MED_C, cfg.LR_FAST_C, cfg.LR_MAX)


def mamdani_lr(v) -> float:
    """Single-point Mamdani inference with centroid defuzzification.

    The output universe is upsampled at each clipped MF's crossing points
    (as skfuzzy does) so the centroid is not biased by the lookup grid.
    """
    v = float(np.clip(v, 0, 1))
    a_low, a_med, a_high = mu_low(v), mu_med(v), mu_high(v)
    # Rules: Low -> Fast, Medium -> Medium, High -> Slow (min implication)
    cuts = [(a_high, SLOW_MF), (a_med, MED_MF), (a_low, FAST_MF)]
    new_values = []
    for cut, mf in cuts:
        if cut > 0.0:
            new_values.extend(_interp_universe(OUT_GRID, mf, cut).tolist())
    universe = np.union1d(OUT_GRID, new_values) if new_values else OUT_GRID
    agg = np.zeros_like(universe)
    for cut, mf in cuts:
        if cut <= 0.0:
            continue
        upsampled = np.interp(universe, OUT_GRID, mf)
        agg = np.maximum(agg, np.minimum(cut, upsampled))  # max aggregation
    if agg.sum() <= 0:
        return float((cfg.LR_SLOW_C + cfg.LR_MED_C + cfg.LR_FAST_C) / 3.0)
    return _centroid_exact(universe, agg)  # centroid defuzz


# Precomputed 201-pt lookup table for training
LR_CURVE = np.array([mamdani_lr(v) for v in V_GRID])
MEAN_LR = float(LR_CURVE.mean())


def fuzzy_lr(v) -> float:
    """Map vol_rank in [0,1] -> LR via linear interp on the precomputed curve."""
    return float(np.interp(np.clip(v, 0, 1), V_GRID, LR_CURVE))


def skfuzzy_reference_curve():
    """Build the identical FIS with skfuzzy.control and evaluate on V_GRID."""
    from skfuzzy import control as ctrl

    vol = ctrl.Antecedent(V_GRID, "vol")
    lr = ctrl.Consequent(OUT_GRID, "lr")
    vol["Low"] = _trimf(V_GRID, 0.0, 0.0, 0.5)
    vol["Medium"] = _trimf(V_GRID, 0.0, 0.5, 1.0)
    vol["High"] = _trimf(V_GRID, 0.5, 1.0, 1.0)
    lr["Slow"] = _trimf(OUT_GRID, 0.0, cfg.LR_SLOW_C, cfg.LR_MED_C)
    lr["Medium"] = _trimf(OUT_GRID, cfg.LR_SLOW_C, cfg.LR_MED_C, cfg.LR_FAST_C)
    lr["Fast"] = _trimf(OUT_GRID, cfg.LR_MED_C, cfg.LR_FAST_C, cfg.LR_MAX)
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
            "surface_min": float(LR_CURVE.min()),
            "surface_max": float(LR_CURVE.max()),
            "vol_min": 0.0, "vol_max": 1.0,
            "smooth_ok": smooth_ok, "mean_ok": mean_ok, "mono_ok": mono_ok,
            "ruspini_maxdiff": ruspini_maxdiff, "ruspini_ok": ruspini_ok,
            "skfuzzy_maxdiff": sk_maxdiff, "skfuzzy_ok": sk_ok,
            "pass": bool(ok)}
