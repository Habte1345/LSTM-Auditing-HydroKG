"""Central registry of all seven rules, and the four-class rollup used for basin-level
dominant-failure-type summaries (per the manuscript's grouping)."""

from __future__ import annotations

from hydrokg.graph.schema import (
    BUDGET_SCALE_FAILURE,
    MAGNITUDE_FAILURE,
    PHYSICAL_IMPOSSIBILITY,
    TIMING_FAILURE,
)
from hydrokg.rules.base import Rule
from hydrokg.rules.r0_negative_flow import NegativeFlowRule
from hydrokg.rules.r1_extreme_ratio import ExtremeRatioRule
from hydrokg.rules.r2_zero_flow_collapse import ZeroFlowCollapseRule
from hydrokg.rules.r3_high_relative_error import HighRelativeErrorRule
from hydrokg.rules.r4_peak_timing import PeakTimingRule
from hydrokg.rules.r5_mass_balance import MassBalanceRule
from hydrokg.rules.r6_budyko import BudykoConsistencyRule

RULE_CLASSES: dict[str, type[Rule]] = {
    "R0": NegativeFlowRule,
    "R1": ExtremeRatioRule,
    "R2": ZeroFlowCollapseRule,
    "R3": HighRelativeErrorRule,
    "R4": PeakTimingRule,
    "R5": MassBalanceRule,
    "R6": BudykoConsistencyRule,
}

# Rules requiring only daily context vs. progressively longer temporal context --
# used by the real-time auditor to stage evaluation (hydrokg/audit/realtime_auditor.py).
DAILY_RULES = ["R0", "R1", "R2", "R3"]
EVENT_RULES = ["R4"]
ANNUAL_RULES = ["R5", "R6"]

VIOLATION_CLASS_TO_RULES = {
    PHYSICAL_IMPOSSIBILITY: ["R0", "R2"],
    MAGNITUDE_FAILURE: ["R1", "R3"],
    TIMING_FAILURE: ["R4"],
    BUDGET_SCALE_FAILURE: ["R5", "R6"],
}


def build_all_rules(**rule_params) -> dict[str, Rule]:
    """Instantiate all seven rules. rule_params: e.g. {'R1': {'low_ratio': 0.15}}."""
    return {
        rule_id: cls(**rule_params.get(rule_id, {}))
        for rule_id, cls in RULE_CLASSES.items()
    }
