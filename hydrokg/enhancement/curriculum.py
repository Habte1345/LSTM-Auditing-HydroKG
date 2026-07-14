"""
Enhancement mechanism 1/3: query-driven curriculum reweighting.

After each training epoch (offline) or streaming window (online), HydroKG is queried for
aggregate violation density by basin and rule (GraphStore.query_violation_hotspots). That
query result -- not a differentiable loss term -- determines how basin-days are weighted
in the *next* training pass: basins/periods with more physical-rule violations are
oversampled, so the model sees more of exactly the conditions it currently gets wrong.
This is deliberately non-differentiable and sits entirely outside the loss function,
distinguishing it from physics-informed-loss (PINN-style) approaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydrokg.graph.base import GraphStore


class ViolationCurriculumSampler:

    def __init__(self, graph_store: GraphStore, temperature: float = 1.0, floor_weight: float = 0.1):
        """
        Parameters
        ----------
        temperature : softens (>1) or sharpens (<1) the weight distribution derived from
            violation counts. temperature=1.0 uses raw normalized counts.
        floor_weight : minimum relative weight given to basins with zero recorded
            violations, so well-behaved basins are never fully excluded from training.
        """
        self.graph = graph_store
        self.temperature = temperature
        self.floor_weight = floor_weight

    def basin_weights(self, all_basin_ids: list[str], top_n: int = 10_000) -> dict[str, float]:
        """Return a normalized sampling weight per basin, derived from the graph's current
        violation hotspot query. Basins absent from the hotspot query (no violations
        recorded) get `floor_weight` (relative to the max observed weight) rather than 0.
        """
        hotspots = self.graph.query_violation_hotspots(top_n=top_n)
        raw = {basin_id: 0.0 for basin_id in all_basin_ids}
        if not hotspots.empty:
            per_basin = hotspots.groupby("basin_id")["count"].sum()
            for basin_id, count in per_basin.items():
                if basin_id in raw:
                    raw[basin_id] = float(count)

        values = np.array(list(raw.values()), dtype=float)
        if values.max() <= 0:
            return {b: 1.0 / len(all_basin_ids) for b in all_basin_ids}

        # temperature-scaled softmax-like normalization over violation counts
        scaled = np.power(values + 1e-9, 1.0 / max(self.temperature, 1e-6))
        floor = self.floor_weight * scaled.max()
        scaled = np.maximum(scaled, floor)
        weights = scaled / scaled.sum()
        return dict(zip(raw.keys(), weights))

    def sample_weights_array(self, basin_ids_per_sample: list[str], all_basin_ids: list[str]) -> np.ndarray:
        """Per-sample weights, suitable for torch.utils.data.WeightedRandomSampler, where
        `basin_ids_per_sample` is the basin id associated with each training example
        (e.g. each (basin, sequence-window) pair in the submodule's CamelsH5 dataset)."""
        weights = self.basin_weights(all_basin_ids)
        return np.array([weights.get(b, self.floor_weight) for b in basin_ids_per_sample])
