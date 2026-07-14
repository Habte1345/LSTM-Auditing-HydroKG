from datetime import date

from hydrokg.graph.memory_store import InMemoryGraphStore
from hydrokg.graph.schema import ViolationRecord


def _make_violation(basin_id, rule_id, day=1, **kwargs):
    return ViolationRecord(
        basin_id=basin_id, rule_id=rule_id, timestamp=date(2020, 1, day),
        q_sim=kwargs.get("q_sim", -1.0), q_obs=kwargs.get("q_obs", 2.0),
        magnitude=kwargs.get("magnitude", -1.0),
        aridity_class=kwargs.get("aridity_class"), landcover_class=kwargs.get("landcover_class"),
    )


def test_write_and_count_violations():
    store = InMemoryGraphStore()
    store.initialize_schema()
    store.register_catchment("A", aridity_class="humid", landcover_class="forest")
    n = store.write_violations([_make_violation("A", "R0"), _make_violation("A", "R0", day=2)])
    assert n == 2
    counts = store.get_violation_counts("A")
    assert counts.loc[counts["rule_id"] == "R0", "count"].iloc[0] == 2


def test_violation_profile_normalizes_to_one():
    store = InMemoryGraphStore()
    store.write_violations([
        _make_violation("A", "R0"), _make_violation("A", "R0", day=2), _make_violation("A", "R1", day=3),
    ])
    profile = store.get_basin_violation_profile("A")
    assert abs(sum(profile.values()) - 1.0) < 1e-9
    assert profile["R0"] > profile["R1"]


def test_query_analog_basins_prefers_class_match_and_low_violations():
    store = InMemoryGraphStore()
    store.register_catchment("target", aridity_class="humid", landcover_class="forest")
    store.register_catchment("good_analog", aridity_class="humid", landcover_class="forest")
    store.register_catchment("bad_analog", aridity_class="humid", landcover_class="forest")
    store.register_catchment("no_match", aridity_class="arid", landcover_class="shrubland")

    store.write_violations([_make_violation("bad_analog", "R0", day=d) for d in range(1, 10)])

    analogs = store.query_analog_basins("target", "R0", "humid", "forest", top_k=3)
    analog_ids = [a for a, _ in analogs]
    assert "no_match" not in analog_ids
    assert analog_ids[0] == "good_analog"  # fewer R0 violations -> ranked first


def test_query_violation_hotspots_weights_sum_to_one():
    store = InMemoryGraphStore()
    store.write_violations([_make_violation("A", "R0", day=d) for d in range(1, 4)]
                            + [_make_violation("B", "R1", day=d) for d in range(1, 2)])
    hotspots = store.query_violation_hotspots()
    assert abs(hotspots["weight"].sum() - 1.0) < 1e-9
    assert hotspots.iloc[0]["basin_id"] == "A"  # more violations -> ranked first
