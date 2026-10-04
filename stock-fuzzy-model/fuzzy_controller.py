"""Fuzzy Logic Controller (soft-computing layer).

Antecedent: batch-average, last-day volatility_14d (Min-Max scaled to [0, 1]).
Consequent: optimizer learning rate.

Rules:
    High volatility   -> Lower learning rate
    Medium volatility -> Medium learning rate
    Low volatility    -> Higher learning rate
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system(vol_p33, vol_p50, vol_p66, show_membership=False):
    """Build the fuzzy control system calibrated to empirical percentiles."""
    # Universes — padded slightly beyond [0, 1] so true extremes never sit
    # exactly on a membership-function endpoint (which would give zero
    # membership everywhere and crash defuzzification).
    volatility_universe = np.linspace(-0.05, 1.05, 111)
    lr_universe = np.linspace(0.0001, 0.01, 100)

    volatility = ctrl.Antecedent(volatility_universe, "volatility")
    learning_rate = ctrl.Consequent(lr_universe, "learning_rate")

    # Low/High use trapmf with a flat "shoulder" so they stay at full
    # membership all the way out to the ends of the universe.
    volatility["Low"] = fuzz.trapmf(volatility.universe, [-0.05, -0.05, vol_p33, vol_p50])
    volatility["Medium"] = fuzz.trimf(volatility.universe, [vol_p33, vol_p50, vol_p66])
    volatility["High"] = fuzz.trapmf(volatility.universe, [vol_p50, vol_p66, 1.05, 1.05])

    learning_rate["Lower"] = fuzz.trimf(learning_rate.universe, [0.0001, 0.0001, 0.005])
    learning_rate["Medium"] = fuzz.trimf(learning_rate.universe, [0.001, 0.005, 0.009])
    learning_rate["Higher"] = fuzz.trimf(learning_rate.universe, [0.005, 0.01, 0.01])

    if show_membership:
        volatility.view()
        learning_rate.view()
        plt.show()

    rule1 = ctrl.Rule(volatility["High"], learning_rate["Lower"])
    rule2 = ctrl.Rule(volatility["Medium"], learning_rate["Medium"])
    rule3 = ctrl.Rule(volatility["Low"], learning_rate["Higher"])

    return ctrl.ControlSystem([rule1, rule2, rule3])


def new_fuzzy_simulator(control_system):
    """scikit-fuzzy simulations are stateful; build a fresh one per query."""
    return ctrl.ControlSystemSimulation(control_system)


def fuzzy_lr(volatility_value, control_system, sim=None, default_lr=1e-3):
    """Map a crisp scaled [0, 1] volatility value to a crisp learning rate.

    Falls back to default_lr instead of crashing if defuzzification has
    no membership to work with (NaN / out-of-range edge case).
    """
    sim = sim or new_fuzzy_simulator(control_system)
    v = float(np.clip(volatility_value, 0.0, 1.0))
    sim.input["volatility"] = v
    try:
        sim.compute()
        return float(sim.output["learning_rate"])
    except (KeyError, ValueError):
        return default_lr
