"""
Offline audit CLI.

Demo mode (no CAMELS, no Neo4j, no submodule run needed):
    python -m hydrokg.cli.run_offline_audit --demo

Real usage:
    python -m hydrokg.cli.run_offline_audit \
        --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run>/lstm_seed<seed>.p \
        --camels_root /path/to/CAMELS_US \
        --graph_backend neo4j --neo4j_uri bolt://localhost:7687 \
        --neo4j_user neo4j --neo4j_password <password>
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from hydrokg.audit.offline_auditor import OfflineAuditor
from hydrokg.data.synthetic import make_synthetic_basin
from hydrokg.evaluation.skill_trust_analysis import summarize_skill_trust
from hydrokg.graph.factory import build_graph_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_demo_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """A handful of synthetic basins with known, controlled violations across all four
    violation classes, plus a couple of clean basins, so the skill-trust scatter shows
    the full range of behavior."""
    basins = {
        "DEMO0001": make_synthetic_basin("DEMO0001", seed=1),  # clean
        "DEMO0002": make_synthetic_basin("DEMO0002", seed=2, inject_negative_flow=True),
        "DEMO0003": make_synthetic_basin("DEMO0003", seed=3, inject_extreme_ratio=True),
        "DEMO0004": make_synthetic_basin("DEMO0004", seed=4, inject_zero_collapse=True),
        "DEMO0005": make_synthetic_basin("DEMO0005", seed=5, inject_peak_lag_days=4),
        "DEMO0006": make_synthetic_basin("DEMO0006", seed=6, inject_mass_balance_violation=True),
        "DEMO0007": make_synthetic_basin("DEMO0007", seed=7),  # clean
    }
    strat = pd.DataFrame({
        "aridity_class": ["humid", "sub_humid", "semi_arid", "humid", "sub_humid", "arid", "humid"],
        "landcover_class": ["forest", "forest", "shrubland", "cropland", "forest", "shrubland", "forest"],
    }, index=list(basins.keys()))
    return basins, strat


def main():
    parser = argparse.ArgumentParser(description="HydroKG offline audit")
    parser.add_argument("--demo", action="store_true", help="Run against synthetic data, no external deps")
    parser.add_argument("--predictions_pickle", type=str, default=None)
    parser.add_argument("--camels_root", type=str, default=None)
    parser.add_argument("--stratification_db", type=str, default=None,
                        help="Path to attributes.db for basin stratification (see submodule's run_dir)")
    parser.add_argument("--graph_backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None, help="Where to write the per-basin audit summary")
    args = parser.parse_args()

    if args.graph_backend == "neo4j":
        graph = build_graph_store("neo4j", uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    else:
        graph = build_graph_store("memory")

    if args.demo:
        basins, stratification = _build_demo_data()
    else:
        if not args.predictions_pickle:
            parser.error("--predictions_pickle is required unless --demo is set")
        from hydrokg.adapters.lstm_adapter import load_predictions_pickle
        from hydrokg.data.basin_attributes import load_basin_stratification
        from hydrokg.data.forcing_loader import attach_precipitation

        basins = load_predictions_pickle(args.predictions_pickle)
        if args.camels_root:
            basins = {
                b: attach_precipitation(df, args.camels_root, b) for b, df in basins.items()
            }
        stratification = (
            load_basin_stratification(args.stratification_db, list(basins.keys()))
            if args.stratification_db else pd.DataFrame(index=list(basins.keys()))
        )

    auditor = OfflineAuditor(graph)
    results = auditor.audit_all(basins, stratification)

    logger.info("Audited %d basins", len(results))
    summary = summarize_skill_trust(results)
    for key, value in summary.items():
        logger.info("%s: %s", key, value)

    if args.output_csv:
        results.to_csv(args.output_csv, index=False)
        logger.info("Wrote per-basin audit summary to %s", args.output_csv)
    else:
        pd.set_option("display.max_columns", None)
        print(results[["basin_id", "kge", "violation_burden", "dominant_class"]])

    graph.close()


if __name__ == "__main__":
    main()
