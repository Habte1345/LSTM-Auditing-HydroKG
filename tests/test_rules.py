"""
Each test injects a KNOWN violation of exactly one rule into synthetic data and checks
that (a) that rule fires on (at least, for stochastic injections) the expected rows and
(b) the rule does not fire at all on clean data (no false positives on well-behaved series).
"""

import numpy as np
import pandas as pd
import pytest

from hydrokg.rules.r0_negative_flow import NegativeFlowRule
from hydrokg.rules.r1_extreme_ratio import ExtremeRatioRule
from hydrokg.rules.r2_zero_flow_collapse import ZeroFlowCollapseRule
from hydrokg.rules.r3_high_relative_error import HighRelativeErrorRule
from hydrokg.rules.r4_peak_timing import PeakTimingRule
from hydrokg.rules.r5_mass_balance import MassBalanceRule
from hydrokg.rules.r6_budyko import BudykoConsistencyRule
from tests.fixtures.synthetic_basin_data import make_synthetic_basin


def test_r0_negative_flow_detects_injected_and_no_false_positives():
    clean = make_synthetic_basin("B_CLEAN", seed=1)
    dirty = make_synthetic_basin("B_DIRTY", seed=1, inject_negative_flow=True)

    rule = NegativeFlowRule()
    assert rule.evaluate("B_CLEAN", clean) == []
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) > 0
    assert all(v.q_sim < 0 for v in violations)
    assert all(v.rule_id == "R0" for v in violations)


def test_r1_extreme_ratio():
    clean = make_synthetic_basin("B_CLEAN", seed=2)
    dirty = make_synthetic_basin("B_DIRTY", seed=2, inject_extreme_ratio=True)

    rule = ExtremeRatioRule()
    assert len(rule.evaluate("B_CLEAN", clean)) == 0
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) > 0
    for v in violations:
        ratio = v.q_sim / v.q_obs
        assert ratio < 0.2 or ratio > 5.0


def test_r2_zero_flow_collapse():
    dirty = make_synthetic_basin("B_DIRTY", seed=3, inject_zero_collapse=True)
    rule = ZeroFlowCollapseRule()
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) > 0
    for v in violations:
        assert v.q_sim < rule.sim_zero_abs
        assert v.q_obs > 0


def test_r3_high_relative_error():
    dirty = make_synthetic_basin("B_DIRTY", seed=4, inject_high_rel_error=True)
    rule = HighRelativeErrorRule(relative_error_threshold=1.0)
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) > 0
    for v in violations:
        assert abs(v.q_sim - v.q_obs) / v.q_obs > 1.0


def test_r4_peak_timing_flags_shifted_peak():
    clean = make_synthetic_basin("B_CLEAN", seed=5)
    shifted = make_synthetic_basin("B_SHIFTED", seed=5, inject_peak_lag_days=5)

    rule = PeakTimingRule(max_lag_days=2)
    shifted_violations = rule.evaluate("B_SHIFTED", shifted)
    assert len(shifted_violations) > 0
    for v in shifted_violations:
        assert v.magnitude > 2


def test_r5_mass_balance_flags_injected_year():
    dirty = make_synthetic_basin("B_DIRTY", seed=6, inject_mass_balance_violation=True)
    clean = make_synthetic_basin("B_CLEAN", seed=6)

    rule = MassBalanceRule()
    assert len(rule.evaluate("B_CLEAN", clean)) == 0
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) >= 1


def test_r6_budyko_consistency_flags_injected_year():
    dirty = make_synthetic_basin("B_DIRTY", seed=7, inject_mass_balance_violation=True)
    rule = BudykoConsistencyRule()
    violations = rule.evaluate("B_DIRTY", dirty)
    assert len(violations) >= 1


def test_n_evaluable_matches_expected_scale():
    df = make_synthetic_basin("B", seed=8, n_years=4)
    assert NegativeFlowRule().n_evaluable(df) == len(df)
    # water-year windows: ~4-5 distinct water years across a 4-year series starting Oct 1
    assert 3 <= PeakTimingRule().n_evaluable(df) <= 5
    assert 3 <= MassBalanceRule().n_evaluable(df) <= 5


@pytest.mark.parametrize("rule_cls", [NegativeFlowRule, ExtremeRatioRule, ZeroFlowCollapseRule,
                                       HighRelativeErrorRule])
def test_daily_rules_never_fire_on_perfect_predictions(rule_cls):
    """A model that predicts observations exactly should never violate any daily rule."""
    df = make_synthetic_basin("B_PERFECT", seed=9)
    df["qsim"] = df["qobs"]
    rule = rule_cls()
    assert rule.evaluate("B_PERFECT", df) == []
