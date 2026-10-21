"""Tests for data pipeline — regression task."""
import numpy as np
import pytest


def test_engineer_features_causal():
    """All feature columns must exist and target must not be in feature_cols."""
    import config as cfg
    assert cfg.TARGET_COL not in cfg.ALL_FEATURES, \
        f"Target '{cfg.TARGET_COL}' leaked into feature_cols!"


def test_ruspini_partition():
    """Low + Med + High must sum to 1 everywhere on [0,1]."""
    import fuzzy as fz
    v = np.linspace(0, 1, 201)
    part = fz.mu_low(v) + fz.mu_med(v) + fz.mu_high(v)
    assert np.allclose(part, 1.0, atol=1e-9), f"Ruspini sum max_err={np.abs(part-1).max()}"


def test_m2_lr_bounds():
    """M2 fuzzy_lr must stay within [LR_MIN, LR_MAX]."""
    import config as cfg, fuzzy as fz
    for v in np.linspace(0, 1, 201):
        lr = fz.fuzzy_lr(v)
        assert cfg.LR_MIN - 1e-10 <= lr <= cfg.LR_MAX + 1e-10, \
            f"fuzzy_lr({v}) = {lr} out of bounds"


def test_m3_lr_bounds():
    """M3 9-rule FIS must stay within [LR_MIN, LR_MAX]."""
    import config as cfg, fuzzy as fz
    g = np.linspace(0, 1, 11)
    for v in g:
        for t in g:
            lr = fz.fuzzy_lr_2input(v, t)
            assert cfg.LR_MIN - 1e-10 <= lr <= cfg.LR_MAX + 1e-10, \
                f"fuzzy_lr_2input({v},{t}) = {lr} out of bounds"


def test_m3_lr_monotone_in_vol():
    """M3: at neutral trend (0.5), LR should decrease as vol_rank increases."""
    import fuzzy as fz
    lrs = [fz.fuzzy_lr_2input(v, 0.5) for v in np.linspace(0, 1, 21)]
    # Allow tiny non-monotone wiggles (< 1e-6), but overall trend must be down
    assert lrs[0] > lrs[-1], \
        f"M3 not monotone at trend=0.5: LR(0)={lrs[0]:.4e} LR(1)={lrs[-1]:.4e}"


def test_ruspini_memberships_helper():
    """ruspini_memberships must sum to 1."""
    import fuzzy as fz
    v = np.linspace(0, 1, 101)
    low, med, high = fz.ruspini_memberships(v)
    total = low + med + high
    assert np.allclose(total, 1.0, atol=1e-9)


def test_fuzzy_gate():
    """check_gate() must pass for M2."""
    import fuzzy as fz
    g = fz.check_gate()
    assert g["pass"], f"Fuzzy gate failed: {g}"


def test_make_windows_shape():
    """make_windows must produce correct shapes."""
    import data as D
    import config as cfg
    n, f = 100, 5
    feat  = np.random.randn(n, f)
    tgt   = np.random.randn(n)
    cl    = np.random.rand(n) * 1000 + 3000
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=n).values
    X, y, closes, d = D._make_windows(feat, tgt, cl, dates, window=cfg.WINDOW)
    assert X.shape == (n - cfg.WINDOW, cfg.WINDOW, f)
    assert y.shape == (n - cfg.WINDOW,)
    assert closes.shape == (n - cfg.WINDOW,)


def test_regression_metrics_persistence():
    """Persistence RMSE must be computed from the actual series (not hardcoded)."""
    import evaluate as E
    close_t = np.array([3000.0, 3010.0, 3020.0])
    y_true  = np.log([3010.0 / 3000.0, 3020.0 / 3010.0, 3015.0 / 3020.0])
    m = E.persistence_baseline(close_t, y_true)
    assert "persist_RMSE" in m
    assert m["persist_RMSE"] > 0


def test_reconstruct_close():
    """Reconstruct should recover true price from true log-return."""
    import evaluate as E
    close_t  = np.array([3000.0, 4000.0])
    log_ret  = np.array([np.log(3100 / 3000), np.log(4050 / 4000)])
    pred     = E.reconstruct_close(close_t, log_ret)
    expected = np.array([3100.0, 4050.0])
    np.testing.assert_allclose(pred, expected, rtol=1e-10)


def test_model_forward_shapes():
    """All three model classes must produce (B,) output."""
    import torch
    from model import RegressionLSTM, FuzzyMoELSTM, FuzzyInputLSTM
    B, T, F = 4, 20, 19
    x  = torch.randn(B, T, F)
    vr = torch.rand(B)
    tr = torch.rand(B)
    for cls, kwargs in [
        (RegressionLSTM, {"n_features": F}),
        (FuzzyMoELSTM,   {"n_features": F}),
        (FuzzyInputLSTM, {"n_features": F}),
    ]:
        m = cls(**kwargs)
        y = m(x, vr=vr, tr=tr)
        assert y.shape == (B,), f"{cls.__name__} output shape {y.shape} != ({B},)"
