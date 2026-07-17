"""
HydroKG's graph layer: schema constants, the abstract GraphStore interface, and both
backend implementations (in-memory dev/test, production Neo4j), plus the factory that
picks between them. Merged into one file (was 6 files) since they're always used
together and none is large on its own.

Canonical schema (entities/relationships as RDF/OWL): src/hydrokg_ontology.ttl.
See docs/ONTOLOGY.md for the full mapping from that ontology to the node labels /
relationship types used here.

Design note: only VIOLATIONS are written as graph facts, never every daily
(prediction, observation, rule-check) triple. At 670 basins x ~30 years x 7 rules,
materializing every check would be ~10^8-10^9 facts, nearly all "rule not violated" --
of no value to curriculum reweighting, analogy correction, or violation embeddings,
all of which only need the violation record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None

# ============================================================================
# Schema constants (mirror src/hydrokg_ontology.ttl 1:1)
# ============================================================================

RULE_IDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]

PHYSICAL_IMPOSSIBILITY = "PhysicalImpossibility"
MAGNITUDE_FAILURE = "MagnitudeFailure"
TIMING_FAILURE = "TimingFailure"
BUDGET_SCALE_FAILURE = "BudgetScaleFailure"

RULE_METADATA = {
    "R0": {"name": "Negative flow", "failure_type": "physical_failure", "violation_class": PHYSICAL_IMPOSSIBILITY, "scale": "daily"},
    "R1": {"name": "Extreme ratio", "failure_type": "predictive_error", "violation_class": MAGNITUDE_FAILURE, "scale": "daily"},
    "R2": {"name": "Zero-flow collapse", "failure_type": "predictive_error", "violation_class": PHYSICAL_IMPOSSIBILITY, "scale": "daily"},
    "R3": {"name": "High relative error", "failure_type": "predictive_error", "violation_class": MAGNITUDE_FAILURE, "scale": "daily"},
    "R4": {"name": "Peak-timing error", "failure_type": "predictive_error", "violation_class": TIMING_FAILURE, "scale": "event"},
    "R5": {"name": "Annual mass balance", "failure_type": "physical_failure", "violation_class": BUDGET_SCALE_FAILURE, "scale": "annual"},
    "R6": {"name": "Budyko consistency", "failure_type": "physical_failure", "violation_class": BUDGET_SCALE_FAILURE, "scale": "annual"},
}

VIOLATION_CLASS_TO_RULES = {
    PHYSICAL_IMPOSSIBILITY: ["R0", "R2"],
    MAGNITUDE_FAILURE: ["R1", "R3"],
    TIMING_FAILURE: ["R4"],
    BUDGET_SCALE_FAILURE: ["R5", "R6"],
}


@dataclass(frozen=True)
class ViolationRecord:
    """One detected rule violation -- the unit written to the graph."""

    basin_id: str
    rule_id: str
    timestamp: date
    q_sim: float
    q_obs: float
    magnitude: float
    aridity_class: Optional[str] = None
    landcover_class: Optional[str] = None
    annual_window: Optional[str] = None
    event_window: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def violation_class(self) -> str:
        return RULE_METADATA[self.rule_id]["violation_class"]

    @property
    def failure_type(self) -> str:
        return RULE_METADATA[self.rule_id]["failure_type"]


# ============================================================================
# Abstract interface
# ============================================================================


class GraphStore(ABC):
    """Contract every HydroKG graph backend must satisfy. Rules/audit/enhancement code
    is written against this interface only, never a specific backend, so swapping
    memory <-> neo4j never requires touching rule or enhancement logic."""

    @abstractmethod
    def initialize_schema(self) -> None:
        """Create fixed Rule/ViolationClass nodes, constraints, and indexes."""

    @abstractmethod
    def register_catchment(self, basin_id: str, aridity_class: Optional[str] = None,
                            landcover_class: Optional[str] = None, attributes: Optional[dict] = None) -> None:
        """Create/update a Catchment node with its static stratification classes."""

    @abstractmethod
    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        """Bulk-write violation records. Returns the number written."""

    @abstractmethod
    def set_basin_metrics(self, basin_id: str, kge: Optional[float] = None,
                           violation_burden: Optional[float] = None) -> None:
        """Attach scalar summary metrics (Eq. 3 output, KGE) to a Catchment node."""

    @abstractmethod
    def upsert_analogy_edges(self, basin_id: str, analogs: list[tuple[str, float]]) -> None:
        """Write/update ANALOGOUS_TO edges from basin_id to (analog_basin_id, weight) pairs."""

    @abstractmethod
    def get_violation_counts(self, basin_id: Optional[str] = None) -> pd.DataFrame:
        """Violation counts per (basin, rule)."""

    @abstractmethod
    def get_basin_violation_profile(self, basin_id: str) -> dict:
        """Per-rule violation rate for one basin -- the violation-history embedding."""

    @abstractmethod
    def query_violation_hotspots(self, top_n: int = 50) -> pd.DataFrame:
        """Basins/rules with the highest violation density, for curriculum reweighting."""

    @abstractmethod
    def query_analog_basins(self, basin_id: str, rule_id: str, aridity_class: str,
                             landcover_class: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Basins sharing aridity/land-cover class with low violation rate for rule_id."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying connection/resources."""

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ============================================================================
# In-memory backend (dev/test/lightweight use, no server required)
# ============================================================================


class InMemoryGraphStore(GraphStore):

    def __init__(self):
        self._violations = pd.DataFrame(columns=[
            "basin_id", "rule_id", "timestamp", "q_sim", "q_obs", "magnitude",
            "aridity_class", "landcover_class", "annual_window", "event_window",
        ])
        self._catchments: dict[str, dict] = {}
        self._analogy_edges: dict[str, list[tuple[str, float]]] = {}
        self._basin_metrics: dict[str, dict] = {}
        # Cache of the full (basin_id, rule_id) -> count aggregation. Rebuilt lazily
        # (only on the next read after a write) rather than recomputed on every single
        # get_violation_counts()/query_analog_basins() call. Without this, a correction
        # pass that queries analog basins once per flagged violation re-groups the ENTIRE
        # violations table (potentially millions of rows after several fine-tuning
        # epochs' worth of online detections) once per candidate basin per flagged
        # violation -- a real, measured performance bug, not a hypothetical one.
        self._counts_cache: pd.DataFrame | None = None

    def initialize_schema(self) -> None:
        return None

    def register_catchment(self, basin_id, aridity_class=None, landcover_class=None, attributes=None) -> None:
        self._catchments[basin_id] = {
            "aridity_class": aridity_class,
            "landcover_class": landcover_class,
            "attributes": attributes or {},
        }

    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        rows = [{
            "basin_id": v.basin_id, "rule_id": v.rule_id, "timestamp": v.timestamp,
            "q_sim": v.q_sim, "q_obs": v.q_obs, "magnitude": v.magnitude,
            "aridity_class": v.aridity_class, "landcover_class": v.landcover_class,
            "annual_window": v.annual_window, "event_window": v.event_window,
        } for v in violations]
        if not rows:
            return 0
        new_df = pd.DataFrame(rows)
        if self._violations.empty:
            self._violations = new_df
        else:
            self._violations = pd.concat([self._violations, new_df], ignore_index=True)
        self._counts_cache = None  # invalidate -- next read rebuilds it once, lazily
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

    def _ensure_counts_cache(self) -> pd.DataFrame:
        """Rebuild the (basin_id, rule_id) -> count aggregation only when the underlying
        violations table has actually changed since the last read. This is what turns
        get_violation_counts/query_analog_basins from an O(total violations) full-table
        scan on every single call into an O(1) cache hit after the first call following
        any write -- the difference between fine on a demo and unusable at ~9M rows
        after a few real fine-tuning epochs."""
        if self._counts_cache is None:
            if self._violations.empty:
                self._counts_cache = pd.DataFrame(columns=["basin_id", "rule_id", "count"])
            else:
                self._counts_cache = (
                    self._violations.groupby(["basin_id", "rule_id"], observed=True)
                    .size().reset_index(name="count")
                )
        return self._counts_cache

    def get_violation_counts(self, basin_id: Optional[str] = None) -> pd.DataFrame:
        cache = self._ensure_counts_cache()
        if basin_id is not None:
            return cache[cache["basin_id"] == basin_id].reset_index(drop=True)
        return cache

    def get_basin_violation_profile(self, basin_id: str) -> dict:
        counts = self.get_violation_counts(basin_id)
        profile = {rid: 0.0 for rid in RULE_IDS}
        total = counts["count"].sum() if not counts.empty else 0
        if total > 0:
            for _, row in counts.iterrows():
                profile[row["rule_id"]] = row["count"] / total
        return profile

    def query_violation_hotspots(self, top_n: int = 50) -> pd.DataFrame:
        cache = self._ensure_counts_cache()
        if cache.empty:
            return pd.DataFrame(columns=["basin_id", "rule_id", "count", "weight"])
        agg = cache.sort_values("count", ascending=False).copy()
        agg["weight"] = agg["count"] / agg["count"].sum()
        return agg.head(top_n).reset_index(drop=True)

    def query_analog_basins(self, basin_id, rule_id, aridity_class, landcover_class, top_k=5):
        # One dict build from the small cached aggregate (at most n_basins x 7 rows),
        # not one full-table groupby per candidate basin per call.
        cache = self._ensure_counts_cache()
        rule_counts = cache[cache["rule_id"] == rule_id].set_index("basin_id")["count"]

        candidates = []
        for other_id, meta in self._catchments.items():
            if other_id == basin_id:
                continue
            class_match = ((meta.get("aridity_class") == aridity_class)
                            + (meta.get("landcover_class") == landcover_class))
            if class_match == 0:
                continue
            rule_count = int(rule_counts.get(other_id, 0))
            score = class_match - np.log1p(rule_count)
            candidates.append((other_id, score))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def close(self) -> None:
        return None


# ============================================================================
# Neo4j backend (production; requires a running Neo4j instance -- docker-compose.yml)
# ============================================================================


class Neo4jGraphStore(GraphStore):

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if GraphDatabase is None:
            raise ImportError(
                "The 'neo4j' package is required for Neo4jGraphStore. Install with "
                "`pip install neo4j` or use InMemoryGraphStore for development."
            )
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def initialize_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("CREATE CONSTRAINT catchment_id IF NOT EXISTS "
                        "FOR (c:Catchment) REQUIRE c.basin_id IS UNIQUE")
            session.run("CREATE CONSTRAINT rule_id IF NOT EXISTS "
                        "FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE")
            session.run("CREATE INDEX violation_basin_rule IF NOT EXISTS "
                        "FOR (v:Violation) ON (v.basin_id, v.rule_id)")
            session.run("CREATE INDEX violation_timestamp IF NOT EXISTS "
                        "FOR (v:Violation) ON (v.timestamp)")
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
                CALL { WITH c WITH c WHERE $aridity_class IS NOT NULL
                       MERGE (a:AridityClass {name: $aridity_class}) MERGE (c)-[:HAS_ARIDITY_CLASS]->(a) }
                CALL { WITH c WITH c WHERE $landcover_class IS NOT NULL
                       MERGE (l:LandCoverClass {name: $landcover_class}) MERGE (c)-[:HAS_LANDCOVER_CLASS]->(l) }
                """,
                basin_id=basin_id, attributes=attributes or {},
                aridity_class=aridity_class, landcover_class=landcover_class,
            )

    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        rows = [{
            "basin_id": v.basin_id, "rule_id": v.rule_id, "timestamp": v.timestamp.isoformat(),
            "q_sim": v.q_sim, "q_obs": v.q_obs, "magnitude": v.magnitude,
            "violation_class": v.violation_class, "aridity_class": v.aridity_class,
            "landcover_class": v.landcover_class, "annual_window": v.annual_window,
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
            session.run("MATCH (c:Catchment {basin_id: $basin_id}) SET c += $updates",
                        basin_id=basin_id, updates=updates)

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
            RETURN other.basin_id AS basin_id, (class_match - log(1 + rule_violations)) AS score
            ORDER BY score DESC
            LIMIT $top_k
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, basin_id=basin_id, rule_id=rule_id,
                                  aridity_class=aridity_class, landcover_class=landcover_class, top_k=top_k)
            return [(r["basin_id"], r["score"]) for r in result]

    def close(self) -> None:
        self._driver.close()


# ============================================================================
# Factory
# ============================================================================


def build_graph_store(backend: str = "memory", **kwargs) -> GraphStore:
    """
    Parameters
    ----------
    backend : {"memory", "neo4j"}
    kwargs : passed through to the backend constructor.
        neo4j: uri, user, password, database (optional)
    """
    if backend == "memory":
        return InMemoryGraphStore()
    elif backend == "neo4j":
        required = {"uri", "user", "password"}
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"Neo4j backend requires {required}; missing {missing}")
        return Neo4jGraphStore(**kwargs)
    else:
        raise ValueError(f"Unknown graph backend '{backend}'. Use 'memory' or 'neo4j'.")