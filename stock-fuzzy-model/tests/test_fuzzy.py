"""Tests for fuzzy module (M2, M3, M5 helpers)."""
import numpy as np
import pytest


def test_m2_gate():
    import fuzzy as fz
    g = fz.check_gate()
    assert g["pass"], str(g)


def test_m2_surface_extremes():
    """vol_rank=0 → fastest LR; vol_rank=1 → slowest."""
    import config as cfg, fuzzy as fz
    lr_lo = fz.fuzzy_lr(0.0)
    lr_hi = fz.fuzzy_lr(1.0)
    assert lr_lo > lr_hi, f"LR not decreasing: LR(0)={lr_lo} LR(1)={lr_hi}"
    # Both in bounds
    assert cfg.LR_MIN <= lr_lo <= cfg.LR_MAX
    assert cfg.LR_MIN <= lr_hi <= cfg.LR_MAX


def test_m3_9_rules_produce_distinct_outputs():
    """M3: (Low vol, High trend) should give different LR than (High vol, Low trend)."""
    import fuzzy as fz
    lr_lh = fz.fuzzy_lr_2input(0.1, 0.9)   # low vol, high trend → Fast
    lr_hl = fz.fuzzy_lr_2input(0.9, 0.1)   # high vol, low trend → Slow
    assert lr_lh > lr_hl, \
        f"M3 wrong direction: LR(low,high)={lr_lh:.4e} LR(high,low)={lr_hl:.4e}"


def test_m3_symmetry_neutral_trend():
    """M3: at trend=0.5 (neutral), LR should match M2 approximately."""
    import fuzzy as fz
    for v in [0.1, 0.5, 0.9]:
        lr_m2 = fz.fuzzy_lr(v)
        lr_m3 = fz.fuzzy_lr_2input(v, 0.5)
        # Should be in the same order-of-magnitude range
        assert abs(lr_m2 - lr_m3) / max(lr_m2, 1e-10) < 0.5, \
            f"M3 deviates too far from M2 at vol={v}: m2={lr_m2:.4e} m3={lr_m3:.4e}"


def test_ruspini_sum_to_one():
    """μ_Low + μ_Med + μ_High == 1 on [0,1]."""
    import fuzzy as fz
    v = np.linspace(0, 1, 501)
    s = fz.mu_low(v) + fz.mu_med(v) + fz.mu_high(v)
    assert np.allclose(s, 1.0, atol=1e-9)


def test_m5_membership_bounds():
    """ruspini_memberships must return values in [0,1] summing to 1."""
    import fuzzy as fz
    v = np.random.rand(100)
    lo, me, hi = fz.ruspini_memberships(v)
    assert (lo >= 0).all() and (lo <= 1).all()
    assert (me >= 0).all() and (me <= 1).all()
    assert (hi >= 0).all() and (hi <= 1).all()
    assert np.allclose(lo + me + hi, 1.0, atol=1e-9)


def test_m3_table_shape():
    """Precomputed M3 lookup table must be 21×21."""
    import fuzzy as fz
    assert fz._M3_TABLE.shape == (21, 21), \
        f"M3 table shape {fz._M3_TABLE.shape} != (21,21)"
