import pandas as pd

from hydrokg.audit.offline_auditor import OfflineAuditor
from hydrokg.evaluation.skill_trust_analysis import summarize_skill_trust
from hydrokg.graph.memory_store import InMemoryGraphStore
from tests.fixtures.synthetic_basin_data import make_synthetic_basin


def test_offline_audit_end_to_end():
    basins = {
        "clean": make_synthetic_basin("clean", seed=100),
        "negflow": make_synthetic_basin("negflow", seed=101, inject_negative_flow=True),
    }
    strat = pd.DataFrame(
        {"aridity_class": ["humid", "humid"], "landcover_class": ["forest", "forest"]},
        index=list(basins.keys()),
    )
    graph = InMemoryGraphStore()
    auditor = OfflineAuditor(graph)
    results = auditor.audit_all(basins, strat)

    assert set(results["basin_id"]) == {"clean", "negflow"}
    assert (results["violation_burden"] >= 0).all()
    assert (results["violation_burden"] <= 1).all()

    negflow_row = results[results["basin_id"] == "negflow"].iloc[0]
    assert negflow_row["violation_counts"]["R0"] > 0

    summary = summarize_skill_trust(results)
    assert summary["n_basins"] == 2

    # graph should reflect what audit_all computed
    graph_counts = graph.get_violation_counts("negflow")
    r0_count = graph_counts.loc[graph_counts["rule_id"] == "R0", "count"]
    assert not r0_count.empty and r0_count.iloc[0] > 0
