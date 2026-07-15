"""
Offline audit CLI -- audits a completed LSTM's predictions against R0-R6 and reports
the skill-trust relationship. Real data only; no demo/synthetic mode.

Usage:
    python -m hydrokg.cli.run_offline_audit \
        --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run>/lstm_seed<seed>.p \
        --camels_root /path/to/CAMELS_US \
        --graph_backend memory
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from hydrokg.audit import OfflineAuditor
from hydrokg.evaluation import summarize_skill_trust
from hydrokg.graph import build_graph_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HydroKG offline audit")
    parser.add_argument("--predictions_pickle", type=str, required=True)
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

    from hydrokg.adapters import load_predictions_pickle
    from hydrokg.data import load_basin_stratification, attach_precipitation

    basins = load_predictions_pickle(args.predictions_pickle)
    if args.camels_root:
        basins = {b: attach_precipitation(df, args.camels_root, b) for b, df in basins.items()}
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
