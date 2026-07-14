"""
End-to-end graph-guided enhancement pipeline, combining all three mechanisms:

  1. Curriculum reweighting of training samples (hydrokg.enhancement.curriculum)
  2. Violation-history embeddings concatenated onto static features (violation_embeddings)
  3. Graph-analogy correction applied to the fine-tuned model's raw output (graph_analogy_correction)

This module has NOT been executed end-to-end in the sandbox this was developed in: it
requires (a) a full CAMELS data directory, (b) a completed submodule training run's
checkpoint, and (c) PyTorch/CUDA, none of which were available. The per-mechanism logic
(curriculum weighting, analogy correction, embeddings) IS independently tested against
synthetic data in tests/test_enhancement.py; this module is the integration point that
wires them into the submodule's own Model/CamelsH5 classes and has only been reviewed for
correctness, not run. Treat it as a strong starting point to validate against a real run,
not a black box to trust blindly -- consistent with how the rest of this repo flags
assumptions rather than hiding them.

No physics-informed loss term is added anywhere in this file -- the loss function used
for fine-tuning is the submodule's own NSELoss/MSELoss, unchanged. All three enhancement
mechanisms operate outside the loss: on sample weighting (before the forward pass), on
input features (concatenated before the forward pass), and on the model's output (after
the forward pass). This is the deliberate distinction from physics-informed-loss
approaches per the project's design decision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hydrokg.adapters.lstm_adapter import _ensure_submodule_on_path, load_run_config
from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.graph_analogy_correction import GraphAnalogyCorrector
from hydrokg.enhancement.violation_embeddings import build_embedding_matrix
from hydrokg.graph.base import GraphStore

logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline:

    def __init__(self, graph_store: GraphStore, run_dir: str | Path, camels_root: str | Path):
        self.graph = graph_store
        self.run_dir = Path(run_dir)
        self.camels_root = Path(camels_root)
        self.run_cfg = load_run_config(self.run_dir)

    def build_auxiliary_inputs(self, basin_ids: list[str]) -> pd.DataFrame:
        """Violation-history embeddings ready to concatenate onto CAMELS static attributes."""
        return build_embedding_matrix(self.graph, basin_ids)

    def build_curriculum_sampler(self, temperature: float = 1.0) -> ViolationCurriculumSampler:
        return ViolationCurriculumSampler(self.graph, temperature=temperature)

    def fine_tune(
        self,
        basin_ids: list[str],
        n_epochs: int = 3,
        learning_rate: float = 5e-4,
        device: str = "cpu",
    ):
        """
        Fine-tunes the submodule's pretrained Model for n_epochs using:
          - curriculum-weighted sampling (WeightedRandomSampler) over training basin-days,
          - static features augmented with the 7-dim violation-history embedding,
          - the submodule's own unmodified NSELoss.

        Returns the fine-tuned model's state_dict. Predictions should then be regenerated
        via the submodule's own evaluate_basin() and passed through
        GraphAnalogyCorrector.correct() (see apply_analogy_correction below) as a separate,
        explicit post-processing step -- correction is intentionally NOT baked into the
        forward pass, so its effect stays auditable and reversible.
        """
        _ensure_submodule_on_path()
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler

        from data.datasets import CamelsH5  # noqa: E402 (submodule import)
        from src.main import Model, GLOBAL_SETTINGS  # noqa: E402 (submodule import)

        device_t = torch.device(device)

        input_size_dyn = 5 if (self.run_cfg["no_static"] or not self.run_cfg["concat_static"]) else 32
        # +7 for the violation-history embedding, only meaningful when static features are
        # concatenated at each timestep (concat_static=True); no_static runs skip this.
        if self.run_cfg["concat_static"] and not self.run_cfg["no_static"]:
            input_size_dyn += 7

        model = Model(
            input_size_dyn=input_size_dyn,
            hidden_size=self.run_cfg["hidden_size"],
            initial_forget_bias=self.run_cfg.get("initial_forget_gate_bias", 5),
            dropout=self.run_cfg["dropout"],
            concat_static=self.run_cfg["concat_static"],
            no_static=self.run_cfg["no_static"],
        ).to(device_t)

        weight_file = self.run_dir / f"model_epoch5.pt"
        state_dict = torch.load(weight_file, map_location=device_t)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            logger.warning(
                "Could not load pretrained weights directly (%s) -- likely because the "
                "input_size_dyn changed by +7 for the violation embedding. Loading "
                "everything except the first LSTM input layer and letting it warm-start "
                "from random init for the new embedding dimensions.", exc,
            )
            model.load_state_dict(state_dict, strict=False)

        ds = CamelsH5(
            h5_file=self.run_dir / "data" / "train" / "train_data.h5",
            basins=basin_ids,
            db_path=str(self.run_dir / "attributes.db"),
            concat_static=self.run_cfg["concat_static"],
            cache=True,
            no_static=self.run_cfg["no_static"],
        )

        sampler_helper = self.build_curriculum_sampler()
        basin_weights = sampler_helper.basin_weights(basin_ids)
        # CamelsH5 does not expose a per-sample basin id array publicly; if your fork does,
        # map it here. Otherwise this falls back to uniform sampling with a logged warning.
        if hasattr(ds, "basin_ids_per_sample"):
            per_sample_weights = np.array([
                basin_weights.get(b, sampler_helper.floor_weight) for b in ds.basin_ids_per_sample
            ])
            sampler = WeightedRandomSampler(per_sample_weights, num_samples=len(ds), replacement=True)
            loader = DataLoader(ds, batch_size=self.run_cfg["batch_size"], sampler=sampler)
        else:
            logger.warning(
                "CamelsH5 has no per-sample basin id attribute in this submodule version; "
                "training without curriculum reweighting. Add a `basin_ids_per_sample` "
                "array to the submodule's dataset class to enable it."
            )
            loader = DataLoader(ds, batch_size=self.run_cfg["batch_size"], shuffle=True)

        from Scripts.nseloss import NSELoss  # noqa: E402

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = NSELoss()

        model.train()
        for epoch in range(1, n_epochs + 1):
            for data in loader:
                optimizer.zero_grad()
                x, y, q_stds = data
                x, y, q_stds = x.to(device_t), y.to(device_t), q_stds.to(device_t)
                predictions = model(x)[0]
                loss = loss_func(predictions, y, q_stds)
                loss.backward()
                optimizer.step()
            logger.info("Enhanced fine-tuning epoch %d complete", epoch)

        return model.state_dict()

    def apply_analogy_correction(
        self,
        raw_predictions: dict[str, pd.DataFrame],
        violation_by_basin: dict[str, list[tuple[str, str]]],
        stratification: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Post-processing pass: for every (basin, timestamp) where a daily rule (R0-R3) was
        flagged, replace the raw q_sim with the graph-analogy-corrected value.

        Parameters
        ----------
        violation_by_basin : {basin_id: [(timestamp_iso, rule_id), ...]} -- typically the
            output of an OfflineAuditor/RealtimeAuditor run against `raw_predictions`.
        """
        corrector = GraphAnalogyCorrector(self.graph, raw_predictions)
        corrected = {b: df.copy() for b, df in raw_predictions.items()}

        for basin_id, flagged in violation_by_basin.items():
            if basin_id not in corrected:
                continue
            aridity_class = stratification.loc[basin_id].get("aridity_class") if basin_id in stratification.index else None
            landcover_class = stratification.loc[basin_id].get("landcover_class") if basin_id in stratification.index else None
            for ts_iso, rule_id in flagged:
                if rule_id not in ("R0", "R1", "R2", "R3"):
                    continue  # event/annual rules are not point-corrected, see module docstring
                ts = pd.Timestamp(ts_iso)
                if ts not in corrected[basin_id].index:
                    continue
                raw_val = corrected[basin_id].loc[ts, "qsim"]
                new_val, _info = corrector.correct(
                    basin_id, ts, raw_val, rule_id, aridity_class, landcover_class
                )
                corrected[basin_id].loc[ts, "qsim"] = new_val

        return corrected
