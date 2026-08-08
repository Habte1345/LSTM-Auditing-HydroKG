# HydroKG Ontology

Canonical schema: [`src/hydrokg_ontology.ttl`](../src/hydrokg_ontology.ttl) (OWL/RDF,
Turtle syntax). This document is a readable companion to that file, mapping each ontology
term to its implementation in both graph backends.

## Classes and Neo4j node labels

| Ontology class | Neo4j label | Materialized for | Notes |
|---|---|---|---|
| `hkg:Catchment` | `Catchment` | every basin in the study | unique on `basin_id` |
| `hkg:Rule` | `Rule` | one of R0-R6 | fixed vocabulary, created at `initialize_schema()` |
| `hkg:ViolationClass` | `ViolationClass` | one of four failure classes | fixed vocabulary |
| `hkg:Violation` | `Violation` | each detected violation | not created for non-violating timesteps; see "Graph granularity" in `docs/ARCHITECTURE.md` |
| `hkg:AridityClass` | `AridityClass` | each aridity stratum observed | derived from the CAMELS `aridity` attribute |
| `hkg:LandCoverClass` | `LandCoverClass` | each land-cover stratum observed | derived from the CAMELS `dom_land_cover` attribute |
| `hkg:TimeStep`, `hkg:EventWindow`, `hkg:AnnualWindow` | properties on `Violation`, not separate nodes | — | see note below |

### TimeStep, EventWindow, and AnnualWindow are properties, not nodes

The ontology defines `TimeStep`, `EventWindow`, and `AnnualWindow` as classes, but both
`GraphStore` implementations store this information as properties on the `Violation` node
(`timestamp`, `event_window`, `annual_window`) rather than as separate connected nodes.
Every query used in this study, including violations on a given date or within a given
water year, filters directly on these `Violation` properties. Promoting these to separate
nodes would be necessary only for queries that reason across timesteps as first-class
entities (for example, identifying other basins with a violation on the same date), which
the ontology supports without requiring a schema change.

## Object properties and Neo4j relationship types

| Ontology property | Neo4j relationship | Direction |
|---|---|---|
| `hkg:forCatchment` | `FOR_CATCHMENT` | `(Violation)-[:FOR_CATCHMENT]->(Catchment)` |
| `hkg:hasRule` | `HAS_RULE` | `(Violation)-[:HAS_RULE]->(Rule)` |
| `hkg:violatesRule` | `VIOLATES_RULE` | `(Catchment)-[:VIOLATES_RULE]->(Rule)`, a direct shortcut used by `query_analog_basins` alongside the full `Violation` record |
| `hkg:hasViolationClass` | `HAS_VIOLATION_CLASS` | `(Rule)-[:HAS_VIOLATION_CLASS]->(ViolationClass)` |
| `hkg:hasAridityClass` | `HAS_ARIDITY_CLASS` | `(Catchment)-[:HAS_ARIDITY_CLASS]->(AridityClass)` |
| `hkg:hasLandCoverClass` | `HAS_LANDCOVER_CLASS` | `(Catchment)-[:HAS_LANDCOVER_CLASS]->(LandCoverClass)` |
| `hkg:analogousTo` | `ANALOGOUS_TO` | `(Catchment)-[:ANALOGOUS_TO {weight}]->(Catchment)`, written by `GraphStore.upsert_analogy_edges` |

## Fixed rule vocabulary

Seven `Rule` nodes and four `ViolationClass` nodes are created once, at
`GraphStore.initialize_schema()`. The Python source of truth is
`src/hydrokg_graph.py::RULE_METADATA`; `scripts/init_neo4j_schema.cypher` provides a
standalone Cypher version for manual inspection independent of the Python driver.

## Extending the ontology

To add an additional rule or stratification dimension:

1. Add the new term to `src/hydrokg_ontology.ttl` (source of truth).
2. Add the corresponding entry to `src/hydrokg_graph.py::RULE_METADATA`, or a new
   constants block for a new stratification dimension.
3. Add a new `Rule` subclass in `src/hydrokg_rules.py` and register it in
   `src/hydrokg_rules.py::RULE_CLASSES`.
4. Verify that `hydrokg_ontology.ttl` and `RULE_METADATA` remain consistent, since no
   automated check currently enforces this correspondence.
