import pandas as pd

from hydrokg.evaluation.enhancement_metrics import compute_deltas, enhancement_summary, percent_improved
from hydrokg.graph.memory_store import InMemoryGraphStore
from hydrokg.graph.schema import ViolationRecord
from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.graph_analogy_correction import GraphAnalogyCorrector
from hydrokg.enhancement.violation_embeddings import build_embedding_matrix
from tests.fixtures.synthetic_basin_data import make_synthetic_basin


def test_eq4_5_6_enhancement_metrics():
    baseline = pd.DataFrame({
        "basin_id": ["A", "B", "C"],
        "kge": [0.5, 0.6, 0.7],
        "violation_burden": [0.2, 0.1, 0.05],
    })
    enhanced = pd.DataFrame({
        "basin_id": ["A", "B", "C"],
        "kge": [0.6, 0.55, 0.9],       # A improved, B worsened, C improved
        "violation_burden": [0.1, 0.15, 0.01],  # A improved, B worsened, C improved
    })
    deltas = compute_deltas(baseline, enhanced)
    assert deltas.loc["A", "delta_kge"] > 0
    assert deltas.loc["B", "delta_kge"] < 0
    assert percent_improved(deltas, "delta_kge") == 100.0 * 2 / 3

    summary = enhancement_summary(baseline, enhanced)
    assert summary["n_basins"] == 3
    assert abs(summary["pct_improved_skill"] - (200.0 / 3)) < 1e-6


def test_curriculum_sampler_prioritizes_violating_basins():
    store = InMemoryGraphStore()
    store.write_violations([
        ViolationRecord(basin_id="A", rule_id="R0", timestamp=__import__("datetime").date(2020, 1, i + 1),
                         q_sim=-1.0, q_obs=1.0, magnitude=-1.0)
        for i in range(20)
    ])
    sampler = ViolationCurriculumSampler(store, floor_weight=0.05)
    weights = sampler.basin_weights(["A", "B"])
    assert weights["A"] > weights["B"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_graph_analogy_correction_produces_nonnegative_r0_fix():
    basins = {
        "clean1": make_synthetic_basin("clean1", seed=10),
        "clean2": make_synthetic_basin("clean2", seed=11),
        "dirty": make_synthetic_basin("dirty", seed=12, inject_negative_flow=True),
    }
    store = InMemoryGraphStore()
    for b in basins:
        store.register_catchment(b, aridity_class="humid", landcover_class="forest")

    corrector = GraphAnalogyCorrector(store, basins)
    negative_rows = basins["dirty"][basins["dirty"]["qsim"] < 0]
    assert len(negative_rows) > 0
    for ts, row in negative_rows.iterrows():
        corrected, info = corrector.correct("dirty", ts, row["qsim"], "R0", "humid", "forest")
        assert corrected >= 0
        assert info["method"] in ("graph_analogy", "no_analogs_fallback", "no_temporal_overlap_fallback")


def test_violation_embedding_matrix_shape():
    store = InMemoryGraphStore()
    store.write_violations([
        ViolationRecord(basin_id="A", rule_id="R1", timestamp=__import__("datetime").date(2020, 1, 1),
                         q_sim=10.0, q_obs=1.0, magnitude=10.0),
    ])
    matrix = build_embedding_matrix(store, ["A", "B"])
    assert matrix.shape == (2, 7)
    assert matrix.loc["B"].sum() == 0.0
