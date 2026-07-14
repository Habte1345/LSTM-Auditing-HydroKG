"""
Enhancement demo CLI.

Demonstrates all three graph-guided enhancement mechanisms against synthetic multi-basin
data using the in-memory graph backend: curriculum reweighting, violation-history
embeddings, and graph-analogy correction. This demo does NOT invoke
hydrokg.enhancement.enhanced_training.EnhancedTrainingPipeline.fine_tune(), which requires
a real CAMELS data directory, a completed submodule training checkpoint, and PyTorch --
see that module's docstring. Everything demonstrated here (curriculum weights, analogy
correction, embeddings) is exactly what fine_tune()/apply_analogy_correction() use
internally, just exercised directly against synthetic data so it can be verified without
those external dependencies.

    python -m hydrokg.cli.run_enhanced_training --demo
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from hydrokg.audit.offline_auditor import OfflineAuditor
from hydrokg.data.synthetic import make_synthetic_basin
from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.graph_analogy_correction import GraphAnalogyCorrector
from hydrokg.enhancement.violation_embeddings import build_embedding_matrix
from hydrokg.graph.factory import build_graph_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HydroKG enhancement mechanism demo")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if not args.demo:
        raise SystemExit(
            "Only --demo is supported directly from this CLI; for a real fine-tuning run "
            "use hydrokg.enhancement.enhanced_training.EnhancedTrainingPipeline in your "
            "own script against a real CAMELS root and submodule checkpoint."
        )

    graph = build_graph_store("memory")

    basins = {
        "DEMO0001": make_synthetic_basin("DEMO0001", seed=1),
        "DEMO0002": make_synthetic_basin("DEMO0002", seed=2, inject_negative_flow=True),
        "DEMO0003": make_synthetic_basin("DEMO0003", seed=3, inject_negative_flow=True),
        "DEMO0004": make_synthetic_basin("DEMO0004", seed=4),
    }
    strat = pd.DataFrame({
        "aridity_class": ["humid", "humid", "humid", "humid"],
        "landcover_class": ["forest", "forest", "forest", "forest"],
    }, index=list(basins.keys()))

    auditor = OfflineAuditor(graph)
    results = auditor.audit_all(basins, strat)
    logger.info("Baseline audit:\n%s", results[["basin_id", "kge", "violation_burden"]])

    # 1. Curriculum reweighting
    sampler = ViolationCurriculumSampler(graph)
    weights = sampler.basin_weights(list(basins.keys()))
    logger.info("Curriculum sampling weights (higher = oversampled next epoch): %s", weights)

    # 2. Violation-history embeddings
    embeddings = build_embedding_matrix(graph, list(basins.keys()))
    logger.info("Violation-history embeddings (auxiliary static features):\n%s", embeddings)

    # 3. Graph-analogy correction: fix DEMO0002/DEMO0003's negative-flow violations using
    #    DEMO0001/DEMO0004 (same aridity/land-cover class, no negative-flow violations) as analogs
    corrector = GraphAnalogyCorrector(graph, basins)
    negative_rows = basins["DEMO0002"][basins["DEMO0002"]["qsim"] < 0]
    n_corrected = 0
    for ts, row in negative_rows.iterrows():
        corrected_val, info = corrector.correct(
            "DEMO0002", ts, row["qsim"], "R0", "humid", "forest",
        )
        n_corrected += 1
        if n_corrected <= 3:
            logger.info("Corrected DEMO0002 @ %s: raw=%.3f -> corrected=%.3f (%s)",
                        ts.date(), row["qsim"], corrected_val, info["method"])
    logger.info("Graph-analogy-corrected %d negative-flow violations in DEMO0002", n_corrected)

    graph.close()


if __name__ == "__main__":
    main()
