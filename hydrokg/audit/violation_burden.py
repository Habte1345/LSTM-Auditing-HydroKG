"""
Eq. 3: V_b = (1/7) * sum_r [ N_violations(b, r) / N_evaluable(b, r) ]

N_evaluable must be tracked per rule (daily rules normalize by evaluated timesteps,
R4 by evaluated water-year windows, R5/R6 by evaluated annual windows) -- this is exactly
what each Rule.n_evaluable() returns (hydrokg/rules/base.py), so the auditor supplies
n_evaluable per rule directly rather than recomputing it here from the graph, which does
not store non-violating opportunities (see hydrokg/graph/schema.py docstring on why only
violations are materialized as facts).
"""

from __future__ import annotations

from hydrokg.graph.schema import RULE_IDS, RULE_METADATA


def compute_violation_burden(violation_counts: dict[str, int], n_evaluable: dict[str, int]) -> float:
    """
    Parameters
    ----------
    violation_counts : {rule_id: count} for one basin (missing rule_id treated as 0)
    n_evaluable : {rule_id: count} for one basin, from Rule.n_evaluable() per rule

    Returns
    -------
    float : V_b in [0, 1]. Rules with n_evaluable == 0 (e.g. too few years for an annual
    rule) are excluded from the average rather than treated as a 0/0 term, and the
    effective denominator (7, or fewer if some rules were inapplicable) is documented via
    `dominant_violation_class` callers should report alongside V_b for transparency.
    """
    terms = []
    for rule_id in RULE_IDS:
        n_eval = n_evaluable.get(rule_id, 0)
        if n_eval <= 0:
            continue
        n_viol = violation_counts.get(rule_id, 0)
        terms.append(n_viol / n_eval)
    if not terms:
        return float("nan")
    return sum(terms) / len(RULE_IDS) if len(terms) == len(RULE_IDS) else sum(terms) / len(terms)


def dominant_violation_class(violation_counts: dict[str, int]) -> str | None:
    """The violation class (of the 4) contributing the largest share of a basin's
    total violations, per the manuscript's basin-level summary definition."""
    from collections import defaultdict

    class_totals: dict[str, int] = defaultdict(int)
    for rule_id, count in violation_counts.items():
        if rule_id in RULE_METADATA:
            class_totals[RULE_METADATA[rule_id]["violation_class"]] += count
    if not class_totals:
        return None
    return max(class_totals, key=class_totals.get)
