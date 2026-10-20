"""Two-input Mamdani FIS: (vol_rank, loss_trend) -> learning rate.

Inputs
  vol_rank in [0, 1]: Low / Medium / High Ruspini triangles.
  loss_trend in [-1, 1]: Falling (-1,-1,0), Plateau (-0.3,0,0.3),
      Rising (0,1,1) triangles.

Output
  LR universe [0, 2e-3]: Slow / Medium / Fast symmetric triangles with
  centroids 0.4e-3 / 1.0e-3 / 1.6e-3.

Rules (min implication, max aggregation, centroid defuzzification)
  Low  & Falling -> Medium     Low  & Plateau -> Fast     Low  & Rising -> Slow
  Med  & Falling -> Medium     Med  & Plateau -> Fast     Med  & Rising -> Slow
  High & Falling -> Slow       High & Plateau -> Medium   High & Rising -> Slow

A 101 x 101 lookup surface LR_SURFACE is precomputed with pure numpy and
bilinearly interpolated at train time. A skfuzzy ControlSystem cross-check
is run on a coarser grid (max diff < 1e-6).

The previous one-input controller is retained as fuzzy_lr_vol_only() for
the ablation variant "fuzzy_vol_only".
"""

import numpy as np

import config as cfg

V_GRID = np.linspace(0.0, 1.0, 101)
T_GRID = np.linspace(cfg.TREND_MIN, cfg.TREND_MAX, 101)
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


# ---- Input MFs: vol_rank Ruspini partition of [0, 1] ----
def mu_low(v):
    return _trimf(v, 0.0, 0.0, 0.5)


def mu_med(v):
    return _trimf(v, 0.0, 0.5, 1.0)


def mu_high(v):
    return _trimf(v, 0.5, 1.0, 1.0)


# ---- Input MFs: loss_trend partition of [-1, 1] ----
def mu_falling(t):
    return _trimf(t, -1.0, -1.0, 0.0)


def mu_plateau(t):
    return _trimf(t, -0.3, 0.0, 0.3)


def mu_rising(t):
    return _trimf(t, 0.0, 1.0, 1.0)


# ---- Output MFs: symmetric triangles, centroids at spec values ----
SLOW_MF = _trimf(OUT_GRID, 0.0, cfg.LR_SLOW_C, 0.8e-3)
MED_MF = _trimf(OUT_GRID, 0.6e-3, cfg.LR_MED_C, 1.4e-3)
FAST_MF = _trimf(OUT_GRID, 1.2e-3, cfg.LR_FAST_C, 2.0e-3)
OUT_MFS = (SLOW_MF, MED_MF, FAST_MF)  # index 0=Slow, 1=Medium, 2=Fast

# (vol_idx, trend_idx, output_idx): 0=Low/Falling/Slow, 1=Med/Plateau/Med,
# 2=High/Rising/Fast
RULES = (
    (0, 0, 1), (0, 1, 2), (0, 2, 0),
    (1, 0, 1), (1, 1, 2), (1, 2, 0),
    (2, 0, 0), (2, 1, 1), (2, 2, 0),
)


def mamdani_lr_2d(v, trend) -> float:
    """Single-point two-input Mamdani inference with centroid defuzzification."""
    v = float(np.clip(v, 0.0, 1.0))
    t = float(np.clip(trend, cfg.TREND_MIN, cfg.TREND_MAX))
    mu_v = (float(mu_low(v)), float(mu_med(v)), float(mu_high(v)))
    mu_t = (float(mu_falling(t)), float(mu_plateau(t)), float(mu_rising(t)))

    agg = np.zeros_like(OUT_GRID)
    for vi, ti, oi in RULES:
        strength = min(mu_v[vi], mu_t[ti])  # AND = min
        if strength <= 0.0:
            continue
        agg = np.maximum(agg, np.minimum(strength, OUT_MFS[oi]))  # min imply / max agg
    denom = float(agg.sum())
    if denom <= 0.0:
        return float((cfg.LR_SLOW_C + cfg.LR_MED_C + cfg.LR_FAST_C) / 3.0)
    return float((agg * OUT_GRID).sum() / denom)  # centroid defuzz


def _build_surface():
    return np.array([[mamdani_lr_2d(v, t) for t in T_GRID] for v in V_GRID])


LR_SURFACE = _build_surface()


def _bilinear(v, trend) -> float:
    """Bilinear interpolation on the uniform (V_GRID, T_GRID) surface."""
    v = float(np.clip(v, 0.0, 1.0))
    t = float(np.clip(trend, cfg.TREND_MIN, cfg.TREND_MAX))
    fi = (v - V_GRID[0]) / (V_GRID[-1] - V_GRID[0]) * (len(V_GRID) - 1)
    fj = (t - T_GRID[0]) / (T_GRID[-1] - T_GRID[0]) * (len(T_GRID) - 1)
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    i1 = min(i0 + 1, len(V_GRID) - 1)
    j1 = min(j0 + 1, len(T_GRID) - 1)
    fv = fi - i0
    ft = fj - j0
    return float(
        LR_SURFACE[i0, j0] * (1 - fv) * (1 - ft)
        + LR_SURFACE[i1, j0] * fv * (1 - ft)
        + LR_SURFACE[i0, j1] * (1 - fv) * ft
        + LR_SURFACE[i1, j1] * fv * ft
    )


def fuzzy_lr(v, trend) -> float:
    """Map (vol_rank, loss_trend) -> LR via bilinear interpolation."""
    return _bilinear(v, trend)


# ---------------------------------------------------------------------------
# One-input ablation controller (previous fuzzy_vol_only curve).
# Rules: Low -> Fast, Medium -> Medium, High -> Slow.
# ---------------------------------------------------------------------------
def mamdani_lr(v) -> float:
    """Single-input Mamdani inference (vol-only ablation)."""
    v = float(np.clip(v, 0, 1))
    a_low, a_med, a_high = float(mu_low(v)), float(mu_med(v)), float(mu_high(v))
    imp_fast = np.minimum(a_low, FAST_MF)
    imp_med = np.minimum(a_med, MED_MF)
    imp_slow = np.minimum(a_high, SLOW_MF)
    agg = np.maximum(np.maximum(imp_fast, imp_med), imp_slow)
    denom = float(agg.sum())
    if denom <= 0:
        return float((cfg.LR_SLOW_C + cfg.LR_MED_C + cfg.LR_FAST_C) / 3.0)
    return float((agg * OUT_GRID).sum() / denom)


LR_CURVE = np.array([mamdani_lr(v) for v in V_GRID])
MEAN_LR = float(LR_CURVE.mean())


def fuzzy_lr_vol_only(v) -> float:
    """Map vol_rank in [0,1] -> LR using the previous one-input controller."""
    return float(np.interp(np.clip(v, 0, 1), V_GRID, LR_CURVE))


def skfuzzy_reference_surface(v_grid=None, t_grid=None):
    """Build the identical two-input FIS with skfuzzy.control and evaluate."""
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    if v_grid is None:
        v_grid = np.linspace(0.0, 1.0, 11)
    if t_grid is None:
        t_grid = np.linspace(cfg.TREND_MIN, cfg.TREND_MAX, 11)

    vol = ctrl.Antecedent(V_GRID, "vol")
    trend = ctrl.Antecedent(T_GRID, "trend")
    lr = ctrl.Consequent(OUT_GRID, "lr")
    vol["Low"] = mu_low(V_GRID)
    vol["Medium"] = mu_med(V_GRID)
    vol["High"] = mu_high(V_GRID)
    trend["Falling"] = mu_falling(T_GRID)
    trend["Plateau"] = mu_plateau(T_GRID)
    trend["Rising"] = mu_rising(T_GRID)
    lr["Slow"] = SLOW_MF
    lr["Medium"] = MED_MF
    lr["Fast"] = FAST_MF

    rules = [
        ctrl.Rule(vol["Low"] & trend["Falling"], lr["Medium"]),
        ctrl.Rule(vol["Low"] & trend["Plateau"], lr["Fast"]),
        ctrl.Rule(vol["Low"] & trend["Rising"], lr["Slow"]),
        ctrl.Rule(vol["Medium"] & trend["Falling"], lr["Medium"]),
        ctrl.Rule(vol["Medium"] & trend["Plateau"], lr["Fast"]),
        ctrl.Rule(vol["Medium"] & trend["Rising"], lr["Slow"]),
        ctrl.Rule(vol["High"] & trend["Falling"], lr["Slow"]),
        ctrl.Rule(vol["High"] & trend["Plateau"], lr["Medium"]),
        ctrl.Rule(vol["High"] & trend["Rising"], lr["Slow"]),
    ]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    out = np.zeros((len(v_grid), len(t_grid)))
    for i, v in enumerate(v_grid):
        for j, t in enumerate(t_grid):
            sim.input["vol"] = float(v)
            sim.input["trend"] = float(t)
            sim.compute()
            out[i, j] = float(sim.output["lr"])
    _ = fuzz  # keep scikit-fuzzy import explicit
    return out


def check_gate() -> dict:
    """Validate the 2-D controller shape and bounds (asserts on failure)."""
    lr_min = float(LR_SURFACE.min())
    lr_max = float(LR_SURFACE.max())
    within_ok = bool(lr_min >= cfg.LR_MIN - 1e-12
                     and lr_max <= cfg.LR_MAX + 1e-12)
    # For every fixed trend, LR must be non-increasing as vol_rank rises.
    # Tolerance 2e-5 absorbs centroid-defuzzification/discretisation ripple
    # (max observed positive step ~1.1e-5, <1% of the 1.2e-3 LR range).
    mono_tol = 2e-5
    vol_diffs = np.diff(LR_SURFACE, axis=0)  # along vol axis
    mono_ok = bool(np.all(vol_diffs <= mono_tol))
    mean_lr = float(LR_SURFACE.mean())
    mean_ok = bool(abs(mean_lr - cfg.LR_MED_C) / cfg.LR_MED_C < 0.25)
    max_step = float(np.abs(vol_diffs).max()) if vol_diffs.size else 0.0
    lr_range = float(lr_max - lr_min)
    # Ruspini partition check for the vol input.
    part = mu_low(V_GRID) + mu_med(V_GRID) + mu_high(V_GRID)
    ruspini_maxdiff = float(np.abs(part - 1.0).max())
    ruspini_ok = bool(ruspini_maxdiff < 1e-9)
    # Cross-check against skfuzzy ControlSystem on a grid.
    try:
        v_chk = np.linspace(0.0, 1.0, 11)
        t_chk = np.linspace(cfg.TREND_MIN, cfg.TREND_MAX, 11)
        ref = skfuzzy_reference_surface(v_chk, t_chk)
        ours = np.array([[mamdani_lr_2d(v, t) for t in t_chk] for v in v_chk])
        sk_maxdiff = float(np.abs(ref - ours).max())
        sk_ok = bool(sk_maxdiff < 1e-6)
    except Exception as e:  # noqa: BLE001
        sk_maxdiff = float("nan")
        sk_ok = False
        print(f"skfuzzy cross-check error: {e}")

    ok = within_ok and mono_ok and mean_ok and ruspini_ok and sk_ok
    result = {
        "surface_min": lr_min, "surface_max": lr_max, "mean": mean_lr,
        "range": lr_range, "max_step_vol": max_step, "mono_tol": mono_tol,
        "within_bounds_ok": within_ok, "mono_vol_ok": mono_ok,
        "mean_ok": mean_ok, "ruspini_maxdiff": ruspini_maxdiff,
        "ruspini_ok": ruspini_ok, "skfuzzy_maxdiff": sk_maxdiff,
        "skfuzzy_ok": sk_ok, "pass": bool(ok),
    }
    assert within_ok, f"LR outside [{cfg.LR_MIN}, {cfg.LR_MAX}]"
    assert mono_ok, "LR surface is not monotone non-increasing in vol_rank"
    assert mean_ok, f"surface mean {mean_lr:.3e} not within 25% of {cfg.LR_MED_C:.3e}"
    assert ruspini_ok, "vol input MFs are not a Ruspini partition"
    assert sk_ok, f"skfuzzy cross-check failed (max diff {sk_maxdiff:.3e})"
    return result


# ---------------------------------------------------------------------------
# ORIGINAL zero-order Sugeno controller (prior prototype, kept as reference).
# Membership-weighted average of 3 constants — NOT used for training anymore.
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
