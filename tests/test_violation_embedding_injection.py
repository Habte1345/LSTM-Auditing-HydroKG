"""
Tests the actual mechanism fixed in this session: writing violation-history embeddings
into a COPY of a CAMELS-style attributes.db, exactly as CamelsH5/load_attributes would
read them (see hydrokg/enhancement/violation_embeddings.py's module docstring for why
this replaced the earlier, broken direct-tensor-concatenation approach).
"""

import sqlite3
from datetime import date

import pandas as pd
import pytest

from hydrokg.enhancement.violation_embeddings import (
    EMBEDDING_COLUMNS,
    n_static_features,
    write_embeddings_to_attributes_db,
)
from hydrokg.graph.memory_store import InMemoryGraphStore
from hydrokg.graph.schema import ViolationRecord


def _make_fake_attributes_db(path, basins):
    """Mimics the submodule's own attributes.db schema closely enough for load_attributes
    (table name 'basin_attributes', indexed by 'gauge_id') to read it correctly. Includes
    gauge_lat/gauge_lon because the submodule's load_attributes unconditionally does
    `df.drop(['gauge_lat', 'gauge_lon'], axis=1)` (no errors='ignore'), so a real
    attributes.db always has them."""
    n = len(basins)
    df = pd.DataFrame({
        "gauge_id": basins,
        "gauge_lat": [40.0 + i for i in range(n)],
        "gauge_lon": [-105.0 - i for i in range(n)],
        "p_mean": [2.0 + 0.5 * i for i in range(n)],
        "aridity": [0.8 + 0.3 * i for i in range(n)],
    }).set_index("gauge_id")
    with sqlite3.connect(path) as conn:
        df.to_sql("basin_attributes", conn, if_exists="replace", index=True, index_label="gauge_id")


def test_write_embeddings_adds_expected_columns(tmp_path):
    basins = ["01013500", "01022500", "01030500"]
    source_db = tmp_path / "attributes.db"
    _make_fake_attributes_db(source_db, basins)

    store = InMemoryGraphStore()
    store.write_violations([
        ViolationRecord(basin_id="01013500", rule_id="R0", timestamp=date(2020, 1, 1),
                         q_sim=-1.0, q_obs=2.0, magnitude=-1.0),
        ViolationRecord(basin_id="01013500", rule_id="R1", timestamp=date(2020, 1, 2),
                         q_sim=10.0, q_obs=1.0, magnitude=10.0),
    ])

    target_db = tmp_path / "attributes_augmented.db"
    result_path = write_embeddings_to_attributes_db(source_db, target_db, store, basins)

    assert result_path == target_db
    assert target_db.exists()
    assert source_db.exists()  # original untouched

    with sqlite3.connect(target_db) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")

    for col in EMBEDDING_COLUMNS:
        assert col in df.columns

    # 01013500 had 1 R0 violation out of 2 total -> rate 0.5
    assert abs(df.loc["01013500", "violation_rate_R0"] - 0.5) < 1e-6
    # basins with no recorded violations get 0 (plus negligible jitter if that column had zero std)
    assert df.loc["01022500", "violation_rate_R0"] < 1e-3


def test_original_static_columns_preserved(tmp_path):
    basins = ["01013500", "01022500"]
    source_db = tmp_path / "attributes.db"
    _make_fake_attributes_db(source_db, basins)
    store = InMemoryGraphStore()

    target_db = tmp_path / "aug.db"
    write_embeddings_to_attributes_db(source_db, target_db, store, basins)

    with sqlite3.connect(target_db) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")

    assert "p_mean" in df.columns
    assert "aridity" in df.columns
    assert df.loc["01013500", "aridity"] == 0.8


def test_n_static_features_counts_augmented_columns(tmp_path):
    basins = ["01013500", "01022500", "01030500"]
    source_db = tmp_path / "attributes.db"
    _make_fake_attributes_db(source_db, basins)
    store = InMemoryGraphStore()

    target_db = tmp_path / "aug.db"
    write_embeddings_to_attributes_db(source_db, target_db, store, basins)

    n_before = n_static_features(source_db, basins)
    n_after = n_static_features(target_db, basins)
    assert n_after == n_before + 7  # exactly the 7 violation_rate_* columns added


def test_zero_variance_rule_does_not_produce_nan_after_normalization(tmp_path):
    """No basin has any violations at all -> every embedding column has zero variance ->
    CamelsH5's (df - mean) / std normalization would divide by zero without the jitter guard."""
    basins = ["01013500", "01022500", "01030500"]
    source_db = tmp_path / "attributes.db"
    _make_fake_attributes_db(source_db, basins)
    store = InMemoryGraphStore()  # no violations written at all

    target_db = tmp_path / "aug.db"
    write_embeddings_to_attributes_db(source_db, target_db, store, basins)

    with sqlite3.connect(target_db) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")

    for col in EMBEDDING_COLUMNS:
        std = df[col].std()
        assert std > 0, f"{col} has zero variance; normalization would divide by zero"
