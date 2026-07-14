"""
Real-time audit CLI. Demonstrates staged rule evaluation by streaming a synthetic basin's
predictions one day at a time, exactly as they would arrive from a live LSTM inference loop.

    python -m hydrokg.cli.run_realtime_audit --demo
"""

from __future__ import annotations

import argparse
import logging

from hydrokg.audit.realtime_auditor import RealtimeAuditor
from hydrokg.data.synthetic import make_synthetic_basin
from hydrokg.graph.factory import build_graph_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HydroKG real-time audit")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--graph_backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default=None)
    args = parser.parse_args()

    if args.graph_backend == "neo4j":
        graph = build_graph_store("neo4j", uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    else:
        graph = build_graph_store("memory")

    auditor = RealtimeAuditor(graph)

    if args.demo:
        df = make_synthetic_basin("DEMO_STREAM", seed=42, inject_negative_flow=True,
                                   inject_peak_lag_days=3, inject_mass_balance_violation=True)
        auditor.register_basin("DEMO_STREAM", aridity_class="humid", landcover_class="forest")

        total_daily_violations = 0
        for ts, row in df.iterrows():
            total_daily_violations += auditor.ingest(
                "DEMO_STREAM", ts, q_sim=row["qsim"], q_obs=row["qobs"], p=row["p"],
                aridity_class="humid", landcover_class="forest",
            )
        total_daily_violations += auditor.flush_all()

        logger.info("Streamed %d daily records for DEMO_STREAM", len(df))
        logger.info("Total violations detected (all rules, staged): %d", total_daily_violations)
        counts = graph.get_violation_counts("DEMO_STREAM")
        print(counts)

    graph.close()


if __name__ == "__main__":
    main()
