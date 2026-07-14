"""
Abstract interface for HydroKG's graph backend.

Two implementations exist:
  - hydrokg.graph.neo4j_store.Neo4jGraphStore   (production; requires a running Neo4j instance)
  - hydrokg.graph.memory_store.InMemoryGraphStore (dev/test substitute; no server required)

Everything in hydrokg.rules, hydrokg.audit, and hydrokg.enhancement is written against this
interface only, so swapping backends never requires touching rule or enhancement logic --
this was a hard requirement given we validated the query logic in this sandbox without a
live Neo4j server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from hydrokg.graph.schema import ViolationRecord


class GraphStore(ABC):
    """Contract every HydroKG graph backend must satisfy."""

    # ---- setup -------------------------------------------------------------
    @abstractmethod
    def initialize_schema(self) -> None:
        """Create fixed Rule/ViolationClass nodes, constraints, and indexes."""

    @abstractmethod
    def register_catchment(
        self,
        basin_id: str,
        aridity_class: Optional[str] = None,
        landcover_class: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> None:
        """Create/update a Catchment node with its static stratification classes."""

    # ---- writes --------------------------------------------------------------
    @abstractmethod
    def write_violations(self, violations: Iterable[ViolationRecord]) -> int:
        """Bulk-write violation records. Returns the number written."""

    @abstractmethod
    def set_basin_metrics(self, basin_id: str, kge: Optional[float] = None,
                           violation_burden: Optional[float] = None) -> None:
        """Attach scalar summary metrics (Eq. 3 output, KGE) to a Catchment node."""

    @abstractmethod
    def upsert_analogy_edges(self, basin_id: str, analogs: list[tuple[str, float]]) -> None:
        """Write/update ANALOGOUS_TO edges from basin_id to a list of (analog_basin_id, weight)."""

    # ---- reads used by audit/evaluation --------------------------------------
    @abstractmethod
    def get_violation_counts(self, basin_id: Optional[str] = None) -> "pandas.DataFrame":  # noqa: F821
        """Violation counts per (basin, rule[, aridity_class, landcover_class])."""

    @abstractmethod
    def get_basin_violation_profile(self, basin_id: str) -> dict:
        """Per-rule violation rate for one basin, used as the violation-history embedding."""

    # ---- reads used by enhancement --------------------------------------------
    @abstractmethod
    def query_violation_hotspots(self, top_n: int = 50) -> "pandas.DataFrame":  # noqa: F821
        """Basins/rules/timesteps with the highest recent violation density, for curriculum reweighting."""

    @abstractmethod
    def query_analog_basins(
        self, basin_id: str, rule_id: str, aridity_class: str, landcover_class: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Basins sharing aridity/land-cover class with low violation rate for rule_id, most-similar first."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying connection/resources."""

    # ---- context manager convenience -----------------------------------------
    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
