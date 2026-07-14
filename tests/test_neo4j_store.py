"""
Tests for hydrokg.graph.neo4j_store.Neo4jGraphStore.

These were NOT run during development of this repo -- no Neo4j server or the `neo4j`
package's native dependencies were available in that sandbox. They mirror
tests/test_graph_store_memory.py exactly (same fixtures, same assertions), so passing
here is the real acceptance bar for the Neo4j backend before trusting it in production.

Run with a live Neo4j instance (e.g. `docker compose up -d` from the repo root):
    NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_PASSWORD=<password> pytest tests/test_neo4j_store.py

Without a reachable server, every test in this file is skipped (not failed).
"""

import os
from datetime import date

import pytest

pytest.importorskip("neo4j")

from hydrokg.graph.neo4j_store import Neo4jGraphStore  # noqa: E402
from hydrokg.graph.schema import ViolationRecord  # noqa: E402

NEO4J_URI = os.environ.get("NEO4J_TEST_URI")
NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD")

pytestmark = pytest.mark.skipif(
    not (NEO4J_URI and NEO4J_PASSWORD),
    reason="Set NEO4J_TEST_URI and NEO4J_TEST_PASSWORD to run live Neo4j tests",
)


@pytest.fixture
def store():
    s = Neo4jGraphStore(uri=NEO4J_URI, user="neo4j", password=NEO4J_PASSWORD)
    s.initialize_schema()
    yield s
    # best-effort cleanup so repeated test runs don't accumulate data
    with s._driver.session(database=s._database) as session:
        session.run("MATCH (n) WHERE n:Catchment OR n:Violation DETACH DELETE n")
    s.close()


def test_write_and_count_violations(store):
    store.register_catchment("A", aridity_class="humid", landcover_class="forest")
    n = store.write_violations([
        ViolationRecord(basin_id="A", rule_id="R0", timestamp=date(2020, 1, 1),
                         q_sim=-1.0, q_obs=2.0, magnitude=-1.0),
        ViolationRecord(basin_id="A", rule_id="R0", timestamp=date(2020, 1, 2),
                         q_sim=-1.0, q_obs=2.0, magnitude=-1.0),
    ])
    assert n == 2
    counts = store.get_violation_counts("A")
    assert counts.loc[counts["rule_id"] == "R0", "count"].iloc[0] == 2


def test_query_analog_basins(store):
    store.register_catchment("target", aridity_class="humid", landcover_class="forest")
    store.register_catchment("good_analog", aridity_class="humid", landcover_class="forest")
    store.register_catchment("bad_analog", aridity_class="humid", landcover_class="forest")
    store.write_violations([
        ViolationRecord(basin_id="bad_analog", rule_id="R0", timestamp=date(2020, 1, i + 1),
                         q_sim=-1.0, q_obs=2.0, magnitude=-1.0)
        for i in range(10)
    ])
    analogs = store.query_analog_basins("target", "R0", "humid", "forest", top_k=3)
    analog_ids = [a for a, _ in analogs]
    assert analog_ids[0] == "good_analog"
