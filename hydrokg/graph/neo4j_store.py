"""
Production Neo4j backend for HydroKG.

Requires a running Neo4j instance -- see docker-compose.yml for a one-command local/HPC
deployment. This module could not be executed end-to-end against a live server in the
sandbox this was developed in (no server binary, restricted egress); its Cypher was
designed directly from the query semantics validated against
hydrokg.graph.memory_store.InMemoryGraphStore, which implements an identical interface.
Before relying on this in production, run tests/test_neo4j_store.py with a real Neo4j
instance reachable (it is skipped automatically otherwise -- see its module docstring).

Install:
    pip install neo4j

Connect:
    from hydrokg.graph.neo4j_store import Neo4jGraphStore
    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="...")
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from hydrokg.graph.base import GraphStore
from hydrokg.graph.schema import RULE_IDS, RULE_METADATA, ViolationRecord

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


class Neo4jGraphStore(GraphStore):

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if GraphDatabase is None:
            raise ImportError(
                "The 'neo4j' package is required for Neo4jGraphStore. Install with "
                "`pip install neo4j` or use hydrokg.graph.memory_store.InMemoryGraphStore "
                "for development without a server."
            )
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    # ---- setup -------------------------------------------------------------------
    def initialize_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT catchment_id IF NOT EXISTS "
                "FOR (c:Catchment) REQUIRE c.basin_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT rule_id IF NOT EXISTS "
                "FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX violation_basin_rule IF NOT EXISTS "
                "FOR (v:Violation) ON (v.basin_id, v.rule_id)"
            )
            session.run(
                "CREATE INDEX violation_timestamp IF NOT EXISTS "
                "FOR (v:Violation) ON (v.timestamp)"
            )
            for rule_id in RULE_IDS:
                meta = RULE_METADATA[rule_id]
                session.run(
                    """
                    MERGE (r:Rule {rule_id: $rule_id})
                    SET r.name = $name, r.failure_type = $failure_type
                    MERGE (vc:ViolationClass {name: $violation_class})
                    MERGE (r)-[:HAS_VIOLATION_CLASS]->(vc)
                    """,
                    rule_id=rule_id, name=meta["name"], failure_type=meta["failure_type"],
                    violation_class=meta["violation_class"],
                )

    def register_catchment(self, basin_id, aridity_class=None, landcover_class=None, attributes=None) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (c:Catchment {basin_id: $basin_id})
                SET c += $attributes
                WITH c
                CALL {
                    WITH c
                    WITH c WHERE $aridity_class IS NOT NULL
                    MERGE (a:AridityClass {name: $aridity_class})
                    MERGE (c)-[:HAS_ARIDITY_CLASS]->(a)
                }
                CALL {
                    WITH c
                    WITH c WHERE $landcover_class IS NOT NULL
                    MERGE (l:LandCoverClass {name: $landcover_class})
                    MERGE (c)-[:HAS_LANDCOVER_CLASS]->(l)
                }
                """,
                basin_id=basin_id, attributes=attributes or {},
                aridity_class=aridity_class, landcover_class=landcover_class,
            )

    # ---- writes --------------------------------------------------------------------
    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        rows = [{
            "basin_id": v.basin_id,
            "rule_id": v.rule_id,
            "timestamp": v.timestamp.isoformat(),
            "q_sim": v.q_sim,
            "q_obs": v.q_obs,
            "magnitude": v.magnitude,
            "violation_class": v.violation_class,
            "aridity_class": v.aridity_class,
            "landcover_class": v.landcover_class,
            "annual_window": v.annual_window,
            "event_window": v.event_window,
        } for v in violations]
        if not rows:
            return 0
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Catchment {basin_id: row.basin_id})
                MATCH (r:Rule {rule_id: row.rule_id})
                CREATE (v:Violation {
                    basin_id: row.basin_id, rule_id: row.rule_id, timestamp: row.timestamp,
                    q_sim: row.q_sim, q_obs: row.q_obs, magnitude: row.magnitude,
                    violation_class: row.violation_class,
                    aridity_class: row.aridity_class, landcover_class: row.landcover_class,
                    annual_window: row.annual_window, event_window: row.event_window
                })
                MERGE (v)-[:FOR_CATCHMENT]->(c)
                MERGE (v)-[:HAS_RULE]->(r)
                MERGE (c)-[:VIOLATES_RULE]->(r)
                """,
                rows=rows,
            )
        return len(rows)

    def set_basin_metrics(self, basin_id: str, kge: Optional[float] = None,
                           violation_burden: Optional[float] = None) -> None:
        updates = {}
        if kge is not None:
            updates["kge"] = kge
        if violation_burden is not None:
            updates["violation_burden"] = violation_burden
        if not updates:
            return
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (c:Catchment {basin_id: $basin_id}) SET c += $updates",
                basin_id=basin_id, updates=updates,
            )

    def upsert_analogy_edges(self, basin_id: str, analogs: list[tuple[str, float]]) -> None:
        rows = [{"other_id": other_id, "weight": weight} for other_id, weight in analogs]
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (c:Catchment {basin_id: $basin_id})
                UNWIND $rows AS row
                MATCH (o:Catchment {basin_id: row.other_id})
                MERGE (c)-[e:ANALOGOUS_TO]->(o)
                SET e.weight = row.weight
                """,
                basin_id=basin_id, rows=rows,
            )

    # ---- reads: audit/evaluation -----------------------------------------------------
    def get_violation_counts(self, basin_id: Optional[str] = None) -> pd.DataFrame:
        query = """
            MATCH (v:Violation)
            WHERE $basin_id IS NULL OR v.basin_id = $basin_id
            RETURN v.basin_id AS basin_id, v.rule_id AS rule_id, count(*) AS count
            ORDER BY count DESC
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, basin_id=basin_id)
            return pd.DataFrame([r.data() for r in result])

    def get_basin_violation_profile(self, basin_id: str) -> dict:
        counts = self.get_violation_counts(basin_id)
        profile = {rid: 0.0 for rid in RULE_IDS}
        total = counts["count"].sum() if not counts.empty else 0
        if total > 0:
            for _, row in counts.iterrows():
                profile[row["rule_id"]] = row["count"] / total
        return profile

    # ---- reads: enhancement --------------------------------------------------------------
    def query_violation_hotspots(self, top_n: int = 50) -> pd.DataFrame:
        query = """
            MATCH (v:Violation)
            RETURN v.basin_id AS basin_id, v.rule_id AS rule_id, count(*) AS count
            ORDER BY count DESC
            LIMIT $top_n
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, top_n=top_n)
            df = pd.DataFrame([r.data() for r in result])
        if not df.empty:
            df["weight"] = df["count"] / df["count"].sum()
        return df

    def query_analog_basins(self, basin_id, rule_id, aridity_class, landcover_class, top_k=5):
        """
        Graph traversal for analogy correction: find catchments sharing the same
        AridityClass and/or LandCoverClass node as `basin_id`, ranked by low violation
        count for `rule_id` (fewest violations = most trustworthy analog).
        """
        query = """
            MATCH (target:Catchment {basin_id: $basin_id})
            MATCH (other:Catchment)
            WHERE other.basin_id <> $basin_id
            OPTIONAL MATCH (other)-[:HAS_ARIDITY_CLASS]->(a:AridityClass {name: $aridity_class})
            OPTIONAL MATCH (other)-[:HAS_LANDCOVER_CLASS]->(l:LandCoverClass {name: $landcover_class})
            WITH other, (CASE WHEN a IS NOT NULL THEN 1 ELSE 0 END
                       + CASE WHEN l IS NOT NULL THEN 1 ELSE 0 END) AS class_match
            WHERE class_match > 0
            OPTIONAL MATCH (other)-[:VIOLATES_RULE]->(:Rule {rule_id: $rule_id})<-[:HAS_RULE]-(v:Violation {basin_id: other.basin_id})
            WITH other, class_match, count(v) AS rule_violations
            RETURN other.basin_id AS basin_id,
                   (class_match - log(1 + rule_violations)) AS score
            ORDER BY score DESC
            LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                query, basin_id=basin_id, rule_id=rule_id,
                aridity_class=aridity_class, landcover_class=landcover_class, top_k=top_k,
            )
            return [(r["basin_id"], r["score"]) for r in result]

    def close(self) -> None:
        self._driver.close()
