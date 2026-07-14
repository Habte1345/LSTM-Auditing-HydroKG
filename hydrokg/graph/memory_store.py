"""
In-memory GraphStore substitute.

Implements the exact same query semantics as hydrokg.graph.neo4j_store.Neo4jGraphStore
(same method signatures, same return shapes) using plain Python/pandas, so that:
  (1) rule/audit/enhancement logic can be developed and unit-tested without a Neo4j server,
  (2) the query logic itself (violation hotspots, analog-basin retrieval) can be validated
      before pointing at production Neo4j.

This is NOT meant for production use at full 670-basin x multi-decade scale -- it holds
everything in a pandas DataFrame in process memory. Swap to Neo4jGraphStore for that;
see hydrokg/graph/neo4j_store.py and docker-compose.yml.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from hydrokg.graph.base import GraphStore
from hydrokg.graph.schema import RULE_IDS, RULE_METADATA, ViolationRecord


class InMemoryGraphStore(GraphStore):

    def __init__(self):
        self._violations = pd.DataFrame(columns=[
            "basin_id", "rule_id", "timestamp", "q_sim", "q_obs", "magnitude",
            "aridity_class", "landcover_class", "annual_window", "event_window",
        ])
        self._catchments: dict[str, dict] = {}
        self._analogy_edges: dict[str, list[tuple[str, float]]] = {}
        self._basin_metrics: dict[str, dict] = {}

    # ---- setup ---------------------------------------------------------------
    def initialize_schema(self) -> None:
        # Fixed Rule vocabulary is just RULE_METADATA; nothing to materialize for the
        # in-memory backend, kept as a no-op to mirror the Neo4jGraphStore interface.
        return None

    def register_catchment(self, basin_id, aridity_class=None, landcover_class=None, attributes=None) -> None:
        self._catchments[basin_id] = {
            "aridity_class": aridity_class,
            "landcover_class": landcover_class,
            "attributes": attributes or {},
        }

    # ---- writes ----------------------------------------------------------------
    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        rows = [{
            "basin_id": v.basin_id,
            "rule_id": v.rule_id,
            "timestamp": v.timestamp,
            "q_sim": v.q_sim,
            "q_obs": v.q_obs,
            "magnitude": v.magnitude,
            "aridity_class": v.aridity_class,
            "landcover_class": v.landcover_class,
            "annual_window": v.annual_window,
            "event_window": v.event_window,
        } for v in violations]
        if not rows:
            return 0
        new_df = pd.DataFrame(rows)
        self._violations = pd.concat([self._violations, new_df], ignore_index=True)
        return len(rows)

    def set_basin_metrics(self, basin_id: str, kge: Optional[float] = None,
                           violation_burden: Optional[float] = None) -> None:
        entry = self._basin_metrics.setdefault(basin_id, {})
        if kge is not None:
            entry["kge"] = kge
        if violation_burden is not None:
            entry["violation_burden"] = violation_burden

    def upsert_analogy_edges(self, basin_id: str, analogs: list[tuple[str, float]]) -> None:
        self._analogy_edges[basin_id] = sorted(analogs, key=lambda x: -x[1])

    # ---- reads: audit/evaluation -------------------------------------------------
    def get_violation_counts(self, basin_id: Optional[str] = None) -> pd.DataFrame:
        df = self._violations
        if basin_id is not None:
            df = df[df["basin_id"] == basin_id]
        if df.empty:
            return pd.DataFrame(columns=["basin_id", "rule_id", "count"])
        return (
            df.groupby(["basin_id", "rule_id"], observed=True)
            .size()
            .reset_index(name="count")
        )

    def get_basin_violation_profile(self, basin_id: str) -> dict:
        """Per-rule violation rate for one basin -> the violation-history embedding vector.

        Rate here is a count normalized by the total number of violations recorded for the
        basin across all rules (a within-basin profile shape), which is what the
        graph-analogy correction and violation-embedding features need -- the *absolute*
        normalized-by-opportunity burden (Eq. 3) is computed separately in
        hydrokg.audit.violation_burden using N_evaluable, which this in-memory store does
        not track (that accounting lives with the auditor, which knows how many timesteps
        were evaluated).
        """
        counts = self.get_violation_counts(basin_id)
        profile = {rid: 0.0 for rid in RULE_IDS}
        total = counts["count"].sum() if not counts.empty else 0
        if total > 0:
            for _, row in counts.iterrows():
                profile[row["rule_id"]] = row["count"] / total
        return profile

    # ---- reads: enhancement --------------------------------------------------------
    def query_violation_hotspots(self, top_n: int = 50) -> pd.DataFrame:
        df = self._violations
        if df.empty:
            return pd.DataFrame(columns=["basin_id", "rule_id", "count", "weight"])
        agg = (
            df.groupby(["basin_id", "rule_id"], observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        agg["weight"] = agg["count"] / agg["count"].sum()
        return agg.head(top_n).reset_index(drop=True)

    def query_analog_basins(self, basin_id, rule_id, aridity_class, landcover_class, top_k=5):
        """Rank candidate basins by (a) matching stratification class and (b) low
        violation rate for rule_id -- most similar / most trustworthy first.
        """
        candidates = []
        for other_id, meta in self._catchments.items():
            if other_id == basin_id:
                continue
            class_match = (
                (meta.get("aridity_class") == aridity_class)
                + (meta.get("landcover_class") == landcover_class)
            )
            if class_match == 0:
                continue
            other_counts = self.get_violation_counts(other_id)
            rule_count = 0
            if not other_counts.empty:
                match = other_counts[other_counts["rule_id"] == rule_id]
                rule_count = int(match["count"].sum()) if not match.empty else 0
            # similarity score: class match weight minus violation penalty
            score = class_match - np.log1p(rule_count)
            candidates.append((other_id, score))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def close(self) -> None:
        return None
