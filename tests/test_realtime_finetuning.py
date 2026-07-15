"""
Tests the core claim behind the "real-time" fine-tuning loop: that writing new
violations to the graph mid-training (via _detect_online_violations) actually changes
what the NEXT epoch's curriculum sampler would compute -- i.e. the graph refresh between
epochs is not a no-op.
"""

import numpy as np

from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.enhanced_training import EnhancedTrainingPipeline
from hydrokg.graph.memory_store import InMemoryGraphStore
from hydrokg.rules.registry import DAILY_RULES, build_all_rules

daily_rules = {rid: rule for rid, rule in build_all_rules().items() if rid in DAILY_RULES}


def test_online_detection_shifts_curriculum_weights_for_next_epoch():
    store = InMemoryGraphStore()
    sampler = ViolationCurriculumSampler(store)

    # "Epoch 1": both basins start with equal (uniform) curriculum weight, no violations yet
    weights_before = sampler.basin_weights(["basin_A", "basin_B"])
    assert abs(weights_before["basin_A"] - weights_before["basin_B"]) < 1e-9

    # Simulate epoch 1's training batches: basin_A's raw model output is consistently bad
    # (negative flow), basin_B's is clean -- exactly what _detect_online_violations would
    # produce from real forward-pass output.
    bad_batch = EnhancedTrainingPipeline._detect_online_violations(
        ["basin_A"] * 10, np.full(10, 2.0), np.full(10, -1.0), daily_rules
    )
    clean_batch = EnhancedTrainingPipeline._detect_online_violations(
        ["basin_B"] * 10, np.full(10, 2.0), np.full(10, 2.05), daily_rules
    )
    assert len(bad_batch) > 0
    assert len(clean_batch) == 0
    store.write_violations(bad_batch)
    store.write_violations(clean_batch)

    # "Epoch 2": recomputing curriculum weights from the now-updated graph should favor
    # basin_A -- this IS the mechanism that makes epoch 2 different from epoch 1, not a
    # frozen pre-training snapshot.
    weights_after = sampler.basin_weights(["basin_A", "basin_B"])
    assert weights_after["basin_A"] > weights_after["basin_B"]
    assert weights_after["basin_A"] > weights_before["basin_A"]


def test_violation_embedding_profile_updates_after_online_detection():
    from hydrokg.enhancement.violation_embeddings import basin_violation_embedding

    store = InMemoryGraphStore()
    profile_before = basin_violation_embedding(store, "basin_A")
    assert profile_before.sum() == 0.0

    violations = EnhancedTrainingPipeline._detect_online_violations(
        ["basin_A"] * 5, np.full(5, 2.0), np.full(5, -1.0), daily_rules
    )
    store.write_violations(violations)

    profile_after = basin_violation_embedding(store, "basin_A")
    assert profile_after.sum() > 0.0  # the embedding a subsequent epoch would train on has changed
