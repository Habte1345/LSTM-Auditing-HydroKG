from hydrokg.audit.violation_burden import compute_violation_burden, dominant_violation_class


def test_burden_zero_when_no_violations():
    counts = {r: 0 for r in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]}
    n_eval = {r: 100 for r in counts}
    assert compute_violation_burden(counts, n_eval) == 0.0


def test_burden_matches_manual_eq3_calculation():
    counts = {"R0": 10, "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0}
    n_eval = {"R0": 100, "R1": 100, "R2": 100, "R3": 100, "R4": 10, "R5": 5, "R6": 5}
    burden = compute_violation_burden(counts, n_eval)
    expected = (1 / 7) * (10 / 100)
    assert abs(burden - expected) < 1e-9


def test_burden_excludes_inapplicable_rules_from_denominator():
    # R4-R6 have zero evaluable opportunities (e.g. too few years of data)
    counts = {"R0": 5, "R1": 0}
    n_eval = {"R0": 100, "R1": 100, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0}
    burden = compute_violation_burden(counts, n_eval)
    expected = ((5 / 100) + 0) / 2
    assert abs(burden - expected) < 1e-9


def test_dominant_violation_class():
    counts = {"R0": 10, "R2": 5, "R1": 1, "R3": 1}
    assert dominant_violation_class(counts) == "PhysicalImpossibility"  # R0+R2=15 > R1+R3=2
