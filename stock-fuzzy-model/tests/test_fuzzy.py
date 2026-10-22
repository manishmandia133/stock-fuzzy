"""Tests for the Mamdani LR controller."""
import numpy as np


def test_m2_gate():
    import fuzzy as fz
    g = fz.check_gate()
    assert g["pass"], str(g)


def test_m2_surface_extremes():
    """vol_rank=0 → fastest LR; vol_rank=1 → slowest."""
    import config as cfg
    import fuzzy as fz
    lr_lo = fz.fuzzy_lr(0.0)
    lr_hi = fz.fuzzy_lr(1.0)
    assert lr_lo > lr_hi, f"LR not decreasing: LR(0)={lr_lo} LR(1)={lr_hi}"
    # Both in bounds
    assert cfg.LR_MIN <= lr_lo <= cfg.LR_MAX
    assert cfg.LR_MIN <= lr_hi <= cfg.LR_MAX


def test_ruspini_sum_to_one():
    """μ_Low + μ_Med + μ_High == 1 on [0,1]."""
    import fuzzy as fz
    v = np.linspace(0, 1, 501)
    s = fz.mu_low(v) + fz.mu_med(v) + fz.mu_high(v)
    assert np.allclose(s, 1.0, atol=1e-9)
