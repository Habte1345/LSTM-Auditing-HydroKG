# HydroKG Ontology

Canonical schema: [`src/hydrokg_ontology.ttl`](../src/hydrokg_ontology.ttl)
(OWL/RDF, Turtle syntax). This document is a readable companion to that file and maps each
ontology term to its concrete implementation in both graph backends.

## Classes → Neo4j node labels

| Ontology class | Neo4j label | Materialized for every... | Notes |
|---|---|---|---|
| `hkg:Catchment` | `Catchment` | basin in the study | unique on `basin_id` |
| `hkg:Rule` | `Rule` | one of R0-R6 | fixed vocabulary, created once at `initialize_schema()` |
| `hkg:ViolationClass` | `ViolationClass` | one of the 4 failure classes | fixed vocabulary |
| `hkg:Violation` | `Violation` | **detected violation only** | not created for non-violating timesteps, see `docs/ARCHITECTURE.md` |
| `hkg:AridityClass` | `AridityClass` | distinct aridity stratum observed | derived from CAMELS `aridity` attribute |
| `hkg:LandCoverClass` | `LandCoverClass` | distinct land-cover stratum observed | derived from CAMELS `dom_land_cover` |
| `hkg:TimeStep`, `hkg:EventWindow`, `hkg:AnnualWindow` | *(properties on Violation, not separate nodes)* | — | see "implementation note" below |

### Implementation note: TimeStep/EventWindow/AnnualWindow are properties, not nodes

The ontology declares `TimeStep`, `EventWindow`, and `AnnualWindow` as classes for
conceptual completeness (a `Violation` "has a TimeStep", "is within an AnnualWindow"), but
both `GraphStore` implementations store these as **properties on the `Violation` node**
(`timestamp`, `event_window`, `annual_window` strings) rather than as separate graph nodes
connected by edges. Materializing a `TimeStep` node for every one of ~11,000 calendar days
across the study period would add graph traversal overhead for zero query benefit here —
every query that needs "violations on this date" or "violations in this water year" can
filter directly on the `Violation` node's property. If a future use case needs to *reason
across* timesteps as first-class entities (e.g. "which other basins had a violation on
the exact same day"), promote these to real nodes at that point; the ontology already
supports it without a schema change.

## Object properties → Neo4j relationship types

| Ontology property | Neo4j relationship | Direction |
|---|---|---|
| `hkg:forCatchment` | `FOR_CATCHMENT` | `(Violation)-[:FOR_CATCHMENT]->(Catchment)` |
| `hkg:hasRule` | `HAS_RULE` | `(Violation)-[:HAS_RULE]->(Rule)` |
| `hkg:violatesRule` | `VIOLATES_RULE` | `(Catchment)-[:VIOLATES_RULE]->(Rule)` — a fast-traversal shortcut alongside the full `Violation` node, used by `query_analog_basins` |
| `hkg:hasViolationClass` | `HAS_VIOLATION_CLASS` | `(Rule)-[:HAS_VIOLATION_CLASS]->(ViolationClass)` |
| `hkg:hasAridityClass` | `HAS_ARIDITY_CLASS` | `(Catchment)-[:HAS_ARIDITY_CLASS]->(AridityClass)` |
| `hkg:hasLandCoverClass` | `HAS_LANDCOVER_CLASS` | `(Catchment)-[:HAS_LANDCOVER_CLASS]->(LandCoverClass)` |
| `hkg:analogousTo` | `ANALOGOUS_TO` | `(Catchment)-[:ANALOGOUS_TO {weight}]->(Catchment)`, written by `GraphStore.upsert_analogy_edges` |

## Fixed rule vocabulary

Seven `Rule` nodes and four `ViolationClass` nodes are created once, at
`GraphStore.initialize_schema()` — see `src/hydrokg_graph.py::RULE_METADATA` (the
Python source of truth, checked against the `.ttl` file by
`tests/test_ontology_sync.py`) and `scripts/init_neo4j_schema.cypher` (a standalone Cypher
version for manual inspection independent of the Python driver).

## Extending the ontology

If you add an eighth rule or a new stratification dimension:

1. Add it to `src/hydrokg_ontology.ttl` first (source of truth).
2. Add the matching entry to `src/hydrokg_graph.py::RULE_METADATA` (or a new constants
   block for a new stratification dimension).
3. Add a new `Rule` subclass in `src/hydrokg_rules.py` and register it in
   `src/hydrokg_rules.py::RULE_CLASSES`.
4. Run `tests/test_ontology_sync.py` — it will fail loudly if the two drift apart.
