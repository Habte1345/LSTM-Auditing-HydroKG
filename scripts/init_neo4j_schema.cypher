// Standalone schema initialization, mirroring hydrokg.graph.neo4j_store.Neo4jGraphStore.initialize_schema().
// Run via `cypher-shell` or the Neo4j Browser to inspect/verify the schema independent of
// the Python driver, or to set up the graph before HydroKG's first Python-side write.

CREATE CONSTRAINT catchment_id IF NOT EXISTS
FOR (c:Catchment) REQUIRE c.basin_id IS UNIQUE;

CREATE CONSTRAINT rule_id IF NOT EXISTS
FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE;

CREATE INDEX violation_basin_rule IF NOT EXISTS
FOR (v:Violation) ON (v.basin_id, v.rule_id);

CREATE INDEX violation_timestamp IF NOT EXISTS
FOR (v:Violation) ON (v.timestamp);

// Fixed rule vocabulary (R0-R6), per src/hydrokg_ontology.ttl
MERGE (r0:Rule {rule_id: "R0"}) SET r0.name = "Negative flow", r0.failure_type = "physical_failure"
MERGE (vc0:ViolationClass {name: "PhysicalImpossibility"})
MERGE (r0)-[:HAS_VIOLATION_CLASS]->(vc0);

MERGE (r1:Rule {rule_id: "R1"}) SET r1.name = "Extreme ratio", r1.failure_type = "predictive_error"
MERGE (vc1:ViolationClass {name: "MagnitudeFailure"})
MERGE (r1)-[:HAS_VIOLATION_CLASS]->(vc1);

MERGE (r2:Rule {rule_id: "R2"}) SET r2.name = "Zero-flow collapse", r2.failure_type = "predictive_error"
MERGE (r2)-[:HAS_VIOLATION_CLASS]->(vc0);

MERGE (r3:Rule {rule_id: "R3"}) SET r3.name = "High relative error", r3.failure_type = "predictive_error"
MERGE (r3)-[:HAS_VIOLATION_CLASS]->(vc1);

MERGE (r4:Rule {rule_id: "R4"}) SET r4.name = "Peak-timing error", r4.failure_type = "predictive_error"
MERGE (vc2:ViolationClass {name: "TimingFailure"})
MERGE (r4)-[:HAS_VIOLATION_CLASS]->(vc2);

MERGE (r5:Rule {rule_id: "R5"}) SET r5.name = "Annual mass balance", r5.failure_type = "physical_failure"
MERGE (vc3:ViolationClass {name: "BudgetScaleFailure"})
MERGE (r5)-[:HAS_VIOLATION_CLASS]->(vc3);

MERGE (r6:Rule {rule_id: "R6"}) SET r6.name = "Budyko consistency", r6.failure_type = "physical_failure"
MERGE (r6)-[:HAS_VIOLATION_CLASS]->(vc3);
