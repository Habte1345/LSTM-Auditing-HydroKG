"""Backend selection: one call site, config-driven, so CLI/scripts never import a
specific backend class directly."""

from __future__ import annotations

from hydrokg.graph.base import GraphStore


def build_graph_store(backend: str = "memory", **kwargs) -> GraphStore:
    """
    Parameters
    ----------
    backend : {"memory", "neo4j"}
    kwargs : passed through to the backend constructor.
        neo4j: uri, user, password, database (optional)
    """
    if backend == "memory":
        from hydrokg.graph.memory_store import InMemoryGraphStore
        return InMemoryGraphStore()
    elif backend == "neo4j":
        from hydrokg.graph.neo4j_store import Neo4jGraphStore
        required = {"uri", "user", "password"}
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"Neo4j backend requires {required}; missing {missing}")
        return Neo4jGraphStore(**kwargs)
    else:
        raise ValueError(f"Unknown graph backend '{backend}'. Use 'memory' or 'neo4j'.")
