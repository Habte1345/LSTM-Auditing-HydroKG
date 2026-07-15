"""
End-to-end graph-guided enhancement pipeline, combining all three mechanisms:

  1. Curriculum reweighting of training samples (hydrokg.enhancement.curriculum)
  2. Violation-history embeddings, injected as extra static-attribute columns in a COPY
     of the run's attributes.db (hydrokg.enhancement.violation_embeddings)
  3. Graph-analogy correction applied to the fine-tuned model's raw output
     (hydrokg.enhancement.graph_analogy_correction)

No physics-informed loss term is added anywhere in this file -- the loss function used
for fine-tuning is the submodule's own NSELoss, unchanged, every batch, every epoch.

REAL-TIME / TRAINING-SUPPORT MODE, PRECISELY: mechanisms 1 and 2 are NOT computed once
before training and left frozen. fine_tune() runs an iterative loop:

  for each epoch:
      - curriculum weights (basin sampling probabilities) and the violation-embedding
        attributes.db are rebuilt from the graph's CURRENT state (the baseline audit on
        epoch 1; the PREVIOUS epoch's own online detections from epoch 2 onward)
      - during training, every batch's own forward-pass output is rescaled back to
        physical mm/day (via the submodule's rescale_features) and checked against R0-R3
        directly -- no extra inference pass, purely a detached side channel that never
        touches the loss or backward pass -- and any violation is written to the graph
        IMMEDIATELY, not buffered to end-of-epoch
      - the next epoch then trains against a graph, and therefore a curriculum/embedding,
        that reflects the model's own most recent behavior, not a stale pre-training snapshot

Scope limit, stated explicitly rather than implied: R4 (peak timing) and R5/R6 (annual
mass balance, Budyko) are NOT evaluated inside this loop. They require a full water-year
of calendar-dated observations, which an isolated training sequence window does not
carry. Those three rules remain evaluated only at the coarser before/after audit
granularity (baseline pre-training, final audit post-training) -- "real-time" in this
codebase means R0-R3 during training, not all seven rules.

Mechanism 2 implementation note: the 7-dim violation embedding is written as ordinary
extra columns into a COPY of the run's attributes.db
(violation_embeddings.write_embeddings_to_attributes_db), not concatenated directly onto
the dynamic input tensor. CamelsH5 already reads static attributes per-basin from
`db_path` and z-score-normalizes/concatenates them automatically
(data/datasets.py::CamelsH5) -- so the embedding is included with zero submodule changes.
Because CamelsH5 reads and caches attributes once at construction, refreshing the
embedding's VALUES between epochs requires reconstructing CamelsH5 each epoch (same
column count throughout, so input_size_dyn and the model architecture never change
mid-training -- only what the static features contain does).

train_data.h5 note: this is a large preprocessing artifact the submodule's own training
run builds once and is correctly gitignored -- it will NOT exist in a fresh submodule
clone even though a run's cfg.json/model weights do. fine_tune() rebuilds it on demand
under this pipeline's work_dir (never inside the submodule's own run_dir) if missing.

Pretrained-weight loading note: PyTorch's `load_state_dict(strict=False)` only skips
missing/unexpected KEYS -- it still raises on a key present in both state dicts whose
SHAPE differs, which is exactly what happens to `lstm.weight_ih` (the only parameter
whose shape depends on input_size_dyn; verified against Scripts/lstm.py -- weight_hh,
bias, and fc.weight/bias are all independent of it). This module filters the checkpoint
by shape before loading, so everything except weight_ih warm-starts correctly.

Performance/memory note (real bug fixed here, not just a tuning caveat): an earlier
version of this method reconstructed CamelsH5 fresh every epoch to pick up refreshed
embedding values. CamelsH5 loads the full x/y arrays into memory at construction
(~12 GiB for a 670-basin, 15-year run) -- reconstructing it meant the previous epoch's
array and the new epoch's array were both alive in memory during the moment of
reassignment, which produced a real ArrayMemoryError on a real run. Fixed by
constructing CamelsH5 exactly ONCE for the whole fine-tuning call, and refreshing only
the small static-attribute table between epochs via the same object's own
_load_attributes() method (which never touches the large arrays) -- see the epoch loop
below. This is also faster, not just safer: the large arrays are no longer silently
re-read every epoch.

Status: reviewed against the submodule's actual code; shape-filtered checkpoint loading
and the online violation-detection helper are unit-tested against synthetic
data/models. The full iterative loop has NOT yet completed an end-to-end run against
real CAMELS data in the environment this was developed in (no CAMELS data available
there, no GPU/large-scale test) -- expect to iterate against your real run.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from hydrokg.adapters.lstm_adapter import _ensure_submodule_on_path, load_run_config
from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.graph_analogy_correction import GraphAnalogyCorrector
from hydrokg.enhancement.violation_embeddings import (
    build_embedding_matrix,
    n_static_features,
    write_embeddings_to_attributes_db,
)
from hydrokg.graph.base import GraphStore

logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline:

    def __init__(self, graph_store: GraphStore, run_dir: str | Path, camels_root: str | Path,
                 work_dir: Optional[str | Path] = None):
        """
        Parameters
        ----------
        graph_store : should already have the BASELINE (traditional-LSTM) audit's
            violations written to it -- e.g. by running OfflineAuditor.audit_all() against
            the traditional predictions first. That baseline graph state is exactly what
            seeds curriculum reweighting and the violation embedding for this fine-tuning
            pass.
        work_dir : where to write the attributes.db copy, train_data.h5 (if missing), and
            enhanced outputs. Defaults to `<run_dir>/hydrokg_enhanced/` so nothing is
            written inside the submodule's own (untouched) run directory tree.
        """
        self.graph = graph_store
        self.run_dir = Path(run_dir)
        self.camels_root = Path(camels_root)
        self.run_cfg = load_run_config(self.run_dir)
        self.work_dir = Path(work_dir) if work_dir else self.run_dir / "hydrokg_enhanced"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_curriculum_sampler(self, temperature: float = 1.0) -> ViolationCurriculumSampler:
        return ViolationCurriculumSampler(self.graph, temperature=temperature)

    def _augmented_db_path(self, basin_ids: list[str]) -> Path:
        source_db = self.run_dir / "attributes.db"
        target_db = self.work_dir / "attributes_with_violation_embedding.db"
        return write_embeddings_to_attributes_db(source_db, target_db, self.graph, basin_ids)

    def _ensure_train_h5(self, basin_ids: list[str]) -> Path:
        """Rebuild train_data.h5 under work_dir if it isn't already sitting in the run's
        own data/train (e.g. because it was correctly gitignored and never pushed)."""
        original_h5 = self.run_dir / "data" / "train" / "train_data.h5"
        if original_h5.exists():
            return original_h5

        rebuilt_h5 = self.work_dir / "train_data.h5"
        if rebuilt_h5.exists():
            return rebuilt_h5

        from hydrokg.adapters.lstm_adapter import create_h5_dataset

        train_start = pd.to_datetime(self.run_cfg["train_start"], format="%d%m%Y")
        train_end = pd.to_datetime(self.run_cfg["train_end"], format="%d%m%Y")
        tqdm.write(
            f"[fine_tune] train_data.h5 not found at {original_h5} (gitignored preprocessing "
            f"artifact, not part of the submodule's git history) -- rebuilding it once from "
            f"CAMELS forcing/discharge under {rebuilt_h5}. This re-reads raw text files for "
            f"{len(basin_ids)} basins and only needs to happen once."
        )
        create_h5_dataset(
            camels_root=self.camels_root,
            out_file=rebuilt_h5,
            basins=basin_ids,
            train_start=train_start,
            train_end=train_end,
            seq_length=self.run_cfg["seq_length"],
        )
        return rebuilt_h5

    def fine_tune(
        self,
        basin_ids: list[str],
        n_epochs: int = 3,
        learning_rate: float = 5e-4,
        device: str = "cpu",
    ) -> tuple[dict, Path]:
        """
        Fine-tunes the submodule's pretrained Model for n_epochs, with the graph updated
        DURING training, not just once beforehand:

          epoch loop:
            1. curriculum weights + violation-embedding attributes.db are rebuilt from the
               graph's CURRENT state (baseline audit on epoch 1; refreshed by the previous
               epoch's online detections from epoch 2 onward)
            2. one epoch of training, with R0-R3 (the four rules that need only a single
               timestep's qsim/qobs, no calendar-date window) evaluated directly on every
               batch's own forward-pass output -- rescaled back to physical mm/day via the
               submodule's own rescale_features, purely as a side channel with .detach(),
               never fed into the loss or the backward pass
            3. every detected violation is written to the SAME graph store immediately
               (not batched to end-of-epoch), so it's visible to anything else watching the
               graph while this epoch is still running
            4. at epoch end, the graph now reflects this epoch's actual model behavior, and
               the next epoch's curriculum weights + embeddings are recomputed from it

        R4 (peak timing) and R5/R6 (annual mass balance, Budyko) are NOT evaluated inside
        this loop -- they require a full water-year of calendar-dated observations, which
        an isolated training sequence window doesn't carry. They remain evaluated only in
        the coarser before/after audits (baseline pre-training, final audit post-training).
        This is a real scope limit, not an oversight -- state it in the manuscript rather
        than implying all 7 rules get real-time treatment.

        The loss function itself is the submodule's own unmodified NSELoss throughout --
        nothing about it changes epoch to epoch. Only (a) which basin-days get sampled and
        (b) the static input's violation-embedding values change between epochs.

        Only meaningful when the run used `concat_static=True`; raises otherwise.

        Returns (state_dict, augmented_db_path) -- pass augmented_db_path to
        generate_predictions() so evaluation uses the same static-attribute set training did.
        """
        if self.run_cfg["no_static"]:
            raise ValueError(
                "This run was trained with no_static=True: there is no static-attribute "
                "channel to inject the violation-history embedding into. Re-train with "
                "concat_static=True to use this mechanism, or use only curriculum "
                "reweighting and graph-analogy correction (which don't require it)."
            )

        _ensure_submodule_on_path()
        import torch
        from torch.utils.data import WeightedRandomSampler

        from data.datasets import CamelsH5  # noqa: E402 (submodule import)
        from data.datautils import rescale_features  # noqa: E402
        from src.main import Model  # noqa: E402 (submodule import)
        from Scripts.nseloss import NSELoss  # noqa: E402

        from hydrokg.rules.registry import DAILY_RULES, build_all_rules
        daily_rules = {rid: rule for rid, rule in build_all_rules().items() if rid in DAILY_RULES}

        device_t = torch.device(device)
        tqdm.write("[fine_tune] Step B: locating/rebuilding train_data.h5")
        train_h5 = self._ensure_train_h5(basin_ids)

        # Static feature count is fixed once (adding embedding COLUMNS never changes their
        # count epoch to epoch, only their VALUES do) so input_size_dyn never changes.
        tqdm.write("[fine_tune] Step A: preparing embedding-augmented attributes.db (epoch 1)")
        augmented_db = self._augmented_db_path(basin_ids)
        n_static = n_static_features(augmented_db, basin_ids)
        input_size_dyn = 5 + n_static

        model = Model(
            input_size_dyn=input_size_dyn,
            hidden_size=self.run_cfg["hidden_size"],
            initial_forget_bias=self.run_cfg.get("initial_forget_gate_bias", 5),
            dropout=self.run_cfg["dropout"],
            concat_static=True,
            no_static=False,
        ).to(device_t)

        tqdm.write("[fine_tune] Step C: warm-starting from pretrained checkpoint")
        weight_file = self.run_dir / "model_epoch5.pt"
        checkpoint_state = torch.load(weight_file, map_location=device_t)
        model_state = model.state_dict()
        compatible = {k: v for k, v in checkpoint_state.items()
                      if k in model_state and model_state[k].shape == v.shape}
        skipped = sorted(set(checkpoint_state.keys()) - set(compatible.keys()))
        model_state.update(compatible)
        model.load_state_dict(model_state)
        tqdm.write(
            f"[fine_tune]   warm-started {len(compatible)}/{len(checkpoint_state)} parameters; "
            f"random-init: {skipped or 'none'}"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = NSELoss()
        batch_size = self.run_cfg["batch_size"]

        tqdm.write("[fine_tune] Step D: loading training data (once for the whole run)")
        # CamelsH5 loads the full x/y arrays into memory here (~12 GiB for a 670-basin,
        # 15-year run) -- this happens ONCE, not once per epoch. Only the static
        # attributes (a few KB) get refreshed between epochs, via the same object's own
        # _load_attributes() method below, never by reconstructing CamelsH5. Reconstructing
        # it every epoch previously meant two ~12 GiB arrays (the old epoch's and the new
        # epoch's) being alive simultaneously during the brief moment of reassignment,
        # which is what caused the reported ArrayMemoryError -- and it also meant silently
        # re-reading the identical, never-changing x/y arrays every epoch for nothing.
        ds = CamelsH5(
            h5_file=train_h5, basins=basin_ids, db_path=str(augmented_db),
            concat_static=True, cache=True, no_static=False,
        )

        tqdm.write(f"[fine_tune] Step E: training for {n_epochs} epoch(s), graph refreshed between each")
        model.train()
        for epoch in range(1, n_epochs + 1):
            if epoch > 1:
                tqdm.write(f"[fine_tune] Refreshing attributes.db + curriculum weights from "
                           f"epoch {epoch - 1}'s online detections")
                augmented_db = self._augmented_db_path(basin_ids)  # same columns, refreshed values
                # Re-run CamelsH5's own attribute-loading step against the refreshed db --
                # this only touches ds.df/attribute_means/attribute_stds (tiny), never
                # re-reads the large h5 arrays (ds.x/ds.y/ds.sample_2_basin untouched).
                ds.db_path = str(augmented_db)
                ds._load_attributes()

            sampler_helper = self.build_curriculum_sampler()
            basin_weights = sampler_helper.basin_weights(basin_ids)
            per_sample_weights = np.array([basin_weights.get(b, sampler_helper.floor_weight)
                                            for b in ds.sample_2_basin])
            indices = list(WeightedRandomSampler(per_sample_weights, num_samples=len(ds), replacement=True))

            running_loss = 0.0
            n_batches = (len(indices) + batch_size - 1) // batch_size
            pbar = tqdm(range(n_batches), desc=f"epoch {epoch}/{n_epochs}", unit="batch", leave=False)
            epoch_violations = 0

            for b in pbar:
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                if not batch_idx:
                    continue
                samples = [ds[i] for i in batch_idx]
                x = torch.stack([s[0] for s in samples]).to(device_t)
                y = torch.stack([s[1] for s in samples]).to(device_t)
                q_stds = torch.stack([s[2] for s in samples]).to(device_t)
                batch_basins = [ds.sample_2_basin[i] for i in batch_idx]

                optimizer.zero_grad()
                predictions = model(x)[0]
                loss = loss_func(predictions, y, q_stds)  # unmodified NSELoss, normalized space
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Side channel only, no gradient: rescale this batch's own forward-pass
                # output back to physical mm/day and check R0-R3 directly against it --
                # zero extra forward passes, genuinely concurrent with training.
                with torch.no_grad():
                    q_sim_phys = rescale_features(predictions.detach().cpu().numpy(), "output").flatten()
                    q_obs_phys = rescale_features(y.detach().cpu().numpy(), "output").flatten()
                violations = self._detect_online_violations(
                    batch_basins, q_obs_phys, q_sim_phys, daily_rules
                )
                if violations:
                    self.graph.write_violations(violations)
                    epoch_violations += len(violations)

                pbar.set_postfix(mean_loss=f"{running_loss / (b + 1):.4f}", violations=epoch_violations)

            tqdm.write(
                f"[fine_tune]   epoch {epoch}/{n_epochs} done: mean loss={running_loss / max(n_batches, 1):.4f}, "
                f"{epoch_violations} online R0-R3 violations detected this epoch"
            )

        state_dict_path = self.work_dir / "enhanced_model_state_dict.pt"
        torch.save(model.state_dict(), state_dict_path)
        return model.state_dict(), augmented_db

    @staticmethod
    def _detect_online_violations(batch_basins, q_obs_phys, q_sim_phys, daily_rules) -> list:
        """Group one training batch's (basin, qobs, qsim) triples by basin and run R0-R3
        against each basin's rows. Dates are synthetic (a plain daily range starting
        2000-01-01, reset per basin per batch) since these rules don't use calendar
        context -- only used so Rule.evaluate()'s pandas-datetime-indexed interface is
        satisfied; never used for windowing, so their specific values don't matter."""
        import pandas as pd

        violations = []
        batch_basins = np.array(batch_basins)
        for basin_id in np.unique(batch_basins):
            mask = batch_basins == basin_id
            n = int(mask.sum())
            df = pd.DataFrame(
                {"qobs": q_obs_phys[mask], "qsim": q_sim_phys[mask]},
                index=pd.date_range("2000-01-01", periods=n, freq="D"),
            )
            for rule in daily_rules.values():
                violations.extend(rule.evaluate(str(basin_id), df))
        return violations

    def generate_predictions(
        self,
        state_dict: dict,
        augmented_db_path: str | Path,
        basin_ids: list[str],
        device: str = "cpu",
    ) -> dict[str, pd.DataFrame]:
        """
        Runs the fine-tuned model over the same validation period the submodule's own
        evaluate() uses (GLOBAL_SETTINGS val_start/val_end), mirroring its evaluate_basin()
        logic exactly, but against the fine-tuned weights and the embedding-augmented
        attributes.db. Returns {basin_id: DataFrame(qobs, qsim)}, the same shape/format
        as the submodule's own predictions pickle.
        """
        _ensure_submodule_on_path()
        import torch
        from torch.utils.data import DataLoader

        from data.datasets import CamelsH5, CamelsTXT  # noqa: E402
        from data.datautils import rescale_features  # noqa: E402
        from src.main import Model, GLOBAL_SETTINGS  # noqa: E402

        device_t = torch.device(device)
        n_static = n_static_features(augmented_db_path, basin_ids)
        input_size_dyn = 5 + n_static

        model = Model(
            input_size_dyn=input_size_dyn,
            hidden_size=self.run_cfg["hidden_size"],
            dropout=self.run_cfg["dropout"],
            concat_static=True,
            no_static=False,
        ).to(device_t)
        model.load_state_dict(state_dict)
        model.eval()

        train_h5 = self._ensure_train_h5(basin_ids)
        ds_train = CamelsH5(
            h5_file=train_h5, db_path=str(augmented_db_path), basins=basin_ids, concat_static=True,
        )
        means = ds_train.get_attribute_means()
        stds = ds_train.get_attribute_stds()

        date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
        results: dict[str, pd.DataFrame] = {}
        skipped_basins = []

        for basin in tqdm(basin_ids, desc="generating predictions", unit="basin"):
            try:
                ds_test = CamelsTXT(
                    camels_root=self.camels_root,
                    basin=basin,
                    dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                    is_train=False,
                    seq_length=self.run_cfg["seq_length"],
                    with_attributes=True,
                    attribute_means=means,
                    attribute_stds=stds,
                    concat_static=True,
                    db_path=str(augmented_db_path),
                )
            except Exception as exc:  # noqa: BLE001
                skipped_basins.append((basin, str(exc)))
                continue

            loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=0)
            preds, obs = None, None
            with torch.no_grad():
                for data in loader:
                    x, y = data[0], data[1]
                    x = x.to(device_t)
                    p = model(x)[0]
                    preds = p.detach().cpu() if preds is None else torch.cat((preds, p.detach().cpu()), 0)
                    obs = y.detach().cpu() if obs is None else torch.cat((obs, y.detach().cpu()), 0)

            preds_np = rescale_features(preds.numpy(), variable="output").flatten()
            obs_np = obs.numpy().flatten()
            n = min(len(preds_np), len(obs_np), len(date_range))
            results[basin] = pd.DataFrame(
                {"qobs": obs_np[:n], "qsim": preds_np[:n]}, index=date_range[:n]
            )

        if skipped_basins:
            tqdm.write(f"[generate_predictions] skipped {len(skipped_basins)} basin(s), e.g.: {skipped_basins[:3]}")
        return results

    def apply_analogy_correction(
        self,
        raw_predictions: dict[str, pd.DataFrame],
        violation_by_basin: dict[str, list[tuple[str, str]]],
        stratification: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Post-processing pass: for every (basin, timestamp) where a daily rule (R0-R3) was
        flagged in `raw_predictions`, replace the raw q_sim with the graph-analogy-corrected
        value. Get `violation_by_basin` by running the rules against `raw_predictions` first
        (a SEPARATE graph/auditor instance from the one used to seed curriculum reweighting,
        so this reflects violations remaining AFTER fine-tuning, not the baseline's).

        Parameters
        ----------
        violation_by_basin : {basin_id: [(timestamp_iso, rule_id), ...]}
        """
        corrector = GraphAnalogyCorrector(self.graph, raw_predictions)
        corrected = {b: df.copy() for b, df in raw_predictions.items()}

        n_flagged = sum(len(v) for v in violation_by_basin.values())
        pbar = tqdm(violation_by_basin.items(), desc="graph-analogy correction", unit="basin",
                    total=len(violation_by_basin))
        n_corrected = 0
        for basin_id, flagged in pbar:
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
                n_corrected += 1
            pbar.set_postfix(corrected=n_corrected, of=n_flagged)

        return corrected

    def save_predictions_pickle(self, predictions: dict[str, pd.DataFrame], filename: str = "enhanced_predictions.p") -> Path:
        """Save in the same {basin_id: DataFrame(qobs, qsim)} shape as the submodule's own
        evaluate() output, so it can be passed straight to hydrokg-audit --predictions_pickle."""
        out_path = self.work_dir / filename
        with open(out_path, "wb") as fp:
            pickle.dump(predictions, fp)
        tqdm.write(f"[save_predictions_pickle] saved enhanced predictions to {out_path}")
        return out_path
