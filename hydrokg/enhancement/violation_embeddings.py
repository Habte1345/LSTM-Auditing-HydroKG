"""
Enhancement mechanism 3/3: violation-history embeddings.

Each basin accumulates a per-rule violation-rate profile in the graph
(GraphStore.get_basin_violation_profile). That 7-dimensional vector (one rate per R0-R6) is
exposed here as an auxiliary static feature, usable alongside CAMELS static attributes in
a fine-tuning pass (hydrokg.enhancement.enhanced_training). This is a graph-derived
representation feeding the model as an input, not a loss term -- the model is free to
learn how to use (or ignore) its own violation history, rather than being penalized
directly for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydrokg.graph.base import GraphStore
from hydrokg.graph.schema import RULE_IDS


def basin_violation_embedding(graph_store: GraphStore, basin_id: str) -> np.ndarray:
    """7-dim vector, one violation rate per rule, in fixed RULE_IDS order."""
    profile = graph_store.get_basin_violation_profile(basin_id)
    return np.array([profile.get(rule_id, 0.0) for rule_id in RULE_IDS], dtype=np.float32)


def build_embedding_matrix(graph_store: GraphStore, basin_ids: list[str]) -> pd.DataFrame:
    """DataFrame indexed by basin_id, columns violation_rate_R0 ... violation_rate_R6 --
    ready to concatenate onto a CAMELS static-attribute DataFrame for fine-tuning."""
    rows = {
        basin_id: basin_violation_embedding(graph_store, basin_id)
        for basin_id in basin_ids
    }
    columns = [f"violation_rate_{rule_id}" for rule_id in RULE_IDS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)
