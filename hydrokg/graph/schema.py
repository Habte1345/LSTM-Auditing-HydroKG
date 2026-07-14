"""
Vocabulary shared by every GraphStore backend.

These constants mirror hydrokg/ontology/hydrokg_ontology.ttl 1:1. Keeping them as plain
Python constants (rather than parsing the Turtle file at runtime) keeps the hot path
(millions of violation writes) free of RDF-parsing overhead; the .ttl file remains the
canonical, human-readable schema documentation and is validated against this module by
tests/test_ontology_sync.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ---- Node labels -----------------------------------------------------------
NODE_CATCHMENT = "Catchment"
NODE_TIMESTEP = "TimeStep"
NODE_PREDICTION = "Prediction"
NODE_OBSERVATION = "Observation"
NODE_RULE = "Rule"
NODE_VIOLATION = "Violation"
NODE_VIOLATION_CLASS = "ViolationClass"
NODE_ARIDITY_CLASS = "AridityClass"
NODE_LANDCOVER_CLASS = "LandCoverClass"
NODE_ANNUAL_WINDOW = "AnnualWindow"
NODE_EVENT_WINDOW = "EventWindow"

# ---- Relationship types -----------------------------------------------------
REL_HAS_DISCHARGE = "HAS_DISCHARGE"
REL_FOR_CATCHMENT = "FOR_CATCHMENT"
REL_HAS_TIMESTEP = "HAS_TIMESTEP"
REL_HAS_RULE = "HAS_RULE"
REL_VIOLATES_RULE = "VIOLATES_RULE"
REL_HAS_VIOLATION_CLASS = "HAS_VIOLATION_CLASS"
REL_HAS_ARIDITY_CLASS = "HAS_ARIDITY_CLASS"
REL_HAS_LANDCOVER_CLASS = "HAS_LANDCOVER_CLASS"
REL_WITHIN_ANNUAL_WINDOW = "WITHIN_ANNUAL_WINDOW"
REL_WITHIN_EVENT_WINDOW = "WITHIN_EVENT_WINDOW"
REL_ANALOGOUS_TO = "ANALOGOUS_TO"

# ---- Fixed rule vocabulary ---------------------------------------------------
RULE_IDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]

PHYSICAL_IMPOSSIBILITY = "PhysicalImpossibility"
MAGNITUDE_FAILURE = "MagnitudeFailure"
TIMING_FAILURE = "TimingFailure"
BUDGET_SCALE_FAILURE = "BudgetScaleFailure"

RULE_METADATA = {
    "R0": {"name": "Negative flow", "failure_type": "physical_failure", "violation_class": PHYSICAL_IMPOSSIBILITY, "scale": "daily"},
    "R1": {"name": "Extreme ratio", "failure_type": "predictive_error", "violation_class": MAGNITUDE_FAILURE, "scale": "daily"},
    "R2": {"name": "Zero-flow collapse", "failure_type": "predictive_error", "violation_class": PHYSICAL_IMPOSSIBILITY, "scale": "daily"},
    "R3": {"name": "High relative error", "failure_type": "predictive_error", "violation_class": MAGNITUDE_FAILURE, "scale": "daily"},
    "R4": {"name": "Peak-timing error", "failure_type": "predictive_error", "violation_class": TIMING_FAILURE, "scale": "event"},
    "R5": {"name": "Annual mass balance", "failure_type": "physical_failure", "violation_class": BUDGET_SCALE_FAILURE, "scale": "annual"},
    "R6": {"name": "Budyko consistency", "failure_type": "physical_failure", "violation_class": BUDGET_SCALE_FAILURE, "scale": "annual"},
}


@dataclass(frozen=True)
class ViolationRecord:
    """One detected rule violation. This is the unit written to the graph.

    Deliberately does NOT store every daily prediction as a node/triple -- at 670 basins
    x ~30 years x 7 rules that is >5B candidate facts. Only violations (the exception, not
    the rule) are materialized, which is also what every downstream consumer (curriculum
    reweighting, analogy correction, violation burden) actually needs.
    """

    basin_id: str
    rule_id: str
    timestamp: date
    q_sim: float
    q_obs: float
    magnitude: float
    aridity_class: Optional[str] = None
    landcover_class: Optional[str] = None
    annual_window: Optional[str] = None
    event_window: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def violation_class(self) -> str:
        return RULE_METADATA[self.rule_id]["violation_class"]

    @property
    def failure_type(self) -> str:
        return RULE_METADATA[self.rule_id]["failure_type"]
