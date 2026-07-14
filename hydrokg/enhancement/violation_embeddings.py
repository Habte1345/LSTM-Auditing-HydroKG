"""
Enhancement mechanism 3/3: violation-history embeddings.

Each basin accumulates a per-rule violation-rate profile in the graph
(GraphStore.get_basin_violation_profile). That 7-dimensional vector (one rate per R0-R6) is
exposed here as an auxiliary static feature.

Injection mechanism (important): the submodule's CamelsH5 dataset (data/datasets.py)
already looks up each sample's basin id (`sample_2_basin`) and pulls static attributes
directly from the sqlite database at `db_path`, z-score-normalizing whatever columns
`data.datautils.load_attributes` returns (anything not in its INVALID_ATTR list). That
means the violation-history embedding can be added as ordinary extra columns in a COPY of
attributes.db -- CamelsH5 will include and normalize them automatically, with no changes
to the submodule and no need to hand-track per-sample basin ids ourselves for this
purpose. `write_embeddings_to_attributes_db` below does exactly that; the original
attributes.db (part of the untouched submodule's run output) is never modified.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from hydrokg.graph.base import GraphStore
from hydrokg.graph.schema import RULE_IDS

EMBEDDING_COLUMNS = [f"violation_rate_{rule_id}" for rule_id in RULE_IDS]


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
    return pd.DataFrame.from_dict(rows, orient="index", columns=EMBEDDING_COLUMNS)


def write_embeddings_to_attributes_db(
    source_db_path: str | Path,
    target_db_path: str | Path,
    graph_store: GraphStore,
    basin_ids: list[str],
) -> Path:
    """
    Copy `source_db_path` (the submodule's own attributes.db for a run) to
    `target_db_path`, then add/overwrite the 7 violation_rate_R0..R6 columns for every
    basin in `basin_ids` in the copy's `basin_attributes` table.

    The source is never modified. Point CamelsH5's `db_path` argument at the returned
    path (the copy) instead of the run's original attributes.db to fine-tune with the
    violation-history embedding included as static input features.

    Note: if a rule has zero variance across all basins at fine-tuning time (e.g. no
    basin has any R6 violations yet), CamelsH5's z-score normalization
    (`(df - mean) / std`) would divide by zero for that column. A tiny deterministic
    jitter is added to break ties in that case -- see the zero-std guard below.
    """
    source_db_path = Path(source_db_path)
    target_db_path = Path(target_db_path)
    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_db_path, target_db_path)

    embeddings = build_embedding_matrix(graph_store, basin_ids)

    # Guard against a rule having identical (often all-zero) values across every basin,
    # which would make CamelsH5's (df - mean) / std normalization divide by zero and
    # inject NaNs into training. A deterministic, basin-index-scaled epsilon breaks ties
    # without materially altering any basin's actual violation-rate signal.
    for col in embeddings.columns:
        if embeddings[col].std() == 0:
            jitter = np.arange(len(embeddings)) * 1e-6
            embeddings[col] = embeddings[col].values + jitter

    with sqlite3.connect(target_db_path) as conn:
        existing = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")
        for col in EMBEDDING_COLUMNS:
            existing[col] = embeddings[col].reindex(existing.index).fillna(0.0)
        existing.to_sql("basin_attributes", conn, if_exists="replace", index=True, index_label="gauge_id")

    return target_db_path


def n_static_features(db_path: str | Path, basins: list[str]) -> int:
    """Number of static attribute columns CamelsH5 will actually load for these basins
    from `db_path` (post INVALID_ATTR filtering) -- use this instead of a hardcoded 32 to
    size the model's input_size_dyn correctly once embedding columns have been added."""
    from hydrokg.adapters.lstm_adapter import load_camels_attributes

    df = load_camels_attributes(db_path, basins)
    return df.shape[1]

