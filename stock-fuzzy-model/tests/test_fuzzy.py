"""Unit tests for the fuzzy_dyn controller (run: python -m unittest)."""
import unittest

import numpy as np

import config as cfg
import fuzzy as fz


class TestFuzzyDynFactor(unittest.TestCase):
    def test_factor_bounds(self):
        for dloss in np.linspace(-0.05, 0.05, 201):
            for vol in np.linspace(0.0, 1.0, 21):
                f = fz.fuzzy_dyn_factor(dloss, vol)
                self.assertGreaterEqual(f, 0.4 - 1e-12)
                self.assertLessEqual(f, 1.15 + 1e-12)

    def test_monotone_worse_means_smaller(self):
        # For a fixed vol, increasing dloss (worse) must not increase the factor.
        for vol in np.linspace(0.0, 1.0, 11):
            factors = [fz.fuzzy_dyn_factor(d, vol)
                       for d in np.linspace(-0.05, 0.05, 401)]
            self.assertTrue(all(factors[i] >= factors[i + 1] - 1e-12
                                for i in range(len(factors) - 1)))

    def test_memberships_partition(self):
        for dloss in np.linspace(-0.02, 0.02, 41):
            mu = fz.fuzzy_memberships(dloss, 0.3)
            self.assertAlmostEqual(mu["improving"] + mu["worse"] +
                                   mu["flat"], 1.0, places=12)
            self.assertAlmostEqual(mu["low"] + mu["high"], 1.0, places=12)

    def test_extreme_rule_outputs(self):
        # Fully improving + low vol -> 1.15
        self.assertAlmostEqual(fz.fuzzy_dyn_factor(-0.01, 0.0), 1.15, places=12)
        # Fully worse + high vol -> 0.40
        self.assertAlmostEqual(fz.fuzzy_dyn_factor(0.01, 1.0), 0.40, places=12)


class TestFuzzyDynController(unittest.TestCase):
    def test_lr_bounds_after_many_updates(self):
        ctrl = fz.FuzzyDynController()
        vloss = 0.69
        for epoch in range(1, 201):
            vol = 0.5
            ctrl.update(vloss, vol)
            self.assertGreaterEqual(ctrl.lr, cfg.LR_MIN - 1e-15)
            self.assertLessEqual(ctrl.lr, cfg.LR_MAX + 1e-15)
            vloss *= 0.99  # constant relative improvement

    def test_warmup_is_linear(self):
        ctrl = fz.FuzzyDynController(base_lr=1e-3, warmup_epochs=3)
        lrs = [ctrl.warmup_lr(e) for e in (1, 2, 3)]
        self.assertAlmostEqual(lrs[0], 1e-3 / 3, places=15)
        self.assertAlmostEqual(lrs[1], 2e-3 / 3, places=15)
        self.assertAlmostEqual(lrs[2], 1e-3, places=15)

    def test_improvement_increases_lr(self):
        ctrl = fz.FuzzyDynController()
        ctrl.prev_vloss = 0.70
        ctrl.lr = 1e-3
        new_lr = ctrl.update(0.69, 0.0)
        self.assertGreater(new_lr, 1e-3)

    def test_worse_high_vol_decreases_lr(self):
        ctrl = fz.FuzzyDynController()
        ctrl.prev_vloss = 0.69
        ctrl.lr = 1e-3
        new_lr = ctrl.update(0.70, 1.0)
        self.assertLess(new_lr, 1e-3)


if __name__ == "__main__":
    unittest.main()
