"""
Graph-guided enhancement: the three non-differentiable mechanisms that use HydroKG's
graph to improve the LSTM WITHOUT a physics-informed loss term, plus the pipeline that
orchestrates them against the real submodule LSTM. Merged from 4 files (curriculum.py,
graph_analogy_correction.py, violation_embeddings.py, enhanced_training.py) since
they're only ever used together.

No physics-informed loss term is added anywhere in this file -- the loss function used
for fine-tuning is the submodule's own NSELoss, unchanged, every batch, every epoch.
The three mechanisms operate entirely outside the loss:

  1. ViolationCurriculumSampler  -- basins/days with more violations get resampled more
     often (changes what data the loss sees, not the loss itself)
  2. write_embeddings_to_attributes_db -- each basin's violation-rate profile becomes an
     extra static input feature (changes model input, not the loss)
  3. GraphAnalogyCorrector -- individual flagged predictions get corrected post-hoc using
     structurally similar, low-violation basins (post-processing, after training)

REAL-TIME / TRAINING-SUPPORT MODE, PRECISELY: mechanisms 1 and 2 are NOT computed once
before training and left frozen. EnhancedTrainingPipeline.fine_tune() runs an iterative
loop:

  for each epoch:
      - curriculum weights and the violation-embedding attributes.db are rebuilt from
        the graph's CURRENT state (the baseline audit on epoch 1; the PREVIOUS epoch's
        own online detections from epoch 2 onward)
      - during training, every batch's own forward-pass output is rescaled back to
        physical mm/day and checked against R0-R3 directly -- no extra inference pass,
        purely a detached side channel that never touches the loss or backward pass --
        and any violation is written to the graph IMMEDIATELY, not buffered to
        end-of-epoch
      - the next epoch then trains against a graph, and therefore a curriculum/embedding,
        that reflects the model's own most recent behavior, not a stale pre-training
        snapshot

Scope limit, stated explicitly: R4 (peak timing) and R5/R6 (annual mass balance,
Budyko) are NOT evaluated inside this loop -- they require a full water-year of
calendar-dated observations an isolated training sequence window doesn't carry. Those
three rules remain evaluated only at the coarser before/after audit granularity.
"real-time" in this codebase means R0-R3 during training, not all seven rules.

Memory/correctness note (a real bug, fixed): CamelsH5 (the submodule's dataset class)
loads the full x/y arrays into memory at construction (~12 GiB for a 670-basin, 15-year
run). An earlier version of fine_tune() reconstructed CamelsH5 every epoch to pick up
refreshed embedding values, which meant two ~12 GiB arrays were briefly alive at once
during reassignment -- this produced a real ArrayMemoryError on a real run. Fixed by
constructing CamelsH5 exactly ONCE for the whole call, and refreshing only the small
static-attribute table between epochs via the same object's own _load_attributes()
method (which never touches the large arrays).

Pretrained-weight loading note: PyTorch's `load_state_dict(strict=False)` only skips
missing/unexpected KEYS -- it still raises on a key present in both state dicts whose
SHAPE differs, which is exactly what happens to `lstm.weight_ih` (the only parameter
whose shape depends on input_size_dyn; verified against Scripts/lstm.py -- weight_hh,
bias, and fc.weight/bias are all independent of it). This module filters the checkpoint
by shape before loading, so everything except weight_ih warm-starts correctly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from hydrokg_adapters import _ensure_submodule_on_path, load_run_config
from hydrokg_graph import GraphStore, RULE_IDS

logger = logging.getLogger(__name__)


# ============================================================================
# Mechanism 1: query-driven curriculum reweighting
# ============================================================================


class ViolationCurriculumSampler:
    """
    After each training epoch, HydroKG is queried for aggregate violation density by
    basin (GraphStore.query_violation_hotspots). That query result -- not a
    differentiable loss term -- determines how basin-days are weighted in the next
    training pass: basins with more physical-rule violations are oversampled.
    """

    def __init__(self, graph_store: GraphStore, temperature: float = 1.0, floor_weight: float = 0.1):
        """
        Parameters
        ----------
        temperature : softens (>1) or sharpens (<1) the weight distribution.
        floor_weight : minimum relative weight for basins with zero recorded violations,
            so well-behaved basins are never fully excluded from training.
        """
        self.graph = graph_store
        self.temperature = temperature
        self.floor_weight = floor_weight

    def basin_weights(self, all_basin_ids: list[str], top_n: int = 10_000) -> dict[str, float]:
        """Return a normalized sampling weight per basin, from the graph's current
        violation hotspot query. Basins with no recorded violations get `floor_weight`
        (relative to the max observed weight) rather than 0."""
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

        scaled = np.power(values + 1e-9, 1.0 / max(self.temperature, 1e-6))
        floor = self.floor_weight * scaled.max()
        scaled = np.maximum(scaled, floor)
        weights = scaled / scaled.sum()
        return dict(zip(raw.keys(), weights))

    def sample_weights_array(self, basin_ids_per_sample: list[str], all_basin_ids: list[str]) -> np.ndarray:
        """Per-sample weights, suitable for torch.utils.data.WeightedRandomSampler."""
        weights = self.basin_weights(all_basin_ids)
        return np.array([weights.get(b, self.floor_weight) for b in basin_ids_per_sample])


# ============================================================================
# Mechanism 2: violation-history embeddings (via a copy of attributes.db)
# ============================================================================

EMBEDDING_COLUMNS = [f"violation_rate_{rule_id}" for rule_id in RULE_IDS]


def basin_violation_embedding(graph_store: GraphStore, basin_id: str) -> np.ndarray:
    """7-dim vector, one violation rate per rule, in fixed RULE_IDS order."""
    profile = graph_store.get_basin_violation_profile(basin_id)
    return np.array([profile.get(rule_id, 0.0) for rule_id in RULE_IDS], dtype=np.float32)


def build_embedding_matrix(graph_store: GraphStore, basin_ids: list[str]) -> pd.DataFrame:
    """DataFrame indexed by basin_id, columns violation_rate_R0 ... violation_rate_R6."""
    rows = {basin_id: basin_violation_embedding(graph_store, basin_id) for basin_id in basin_ids}
    return pd.DataFrame.from_dict(rows, orient="index", columns=EMBEDDING_COLUMNS)


def write_embeddings_to_attributes_db(source_db_path: str | Path, target_db_path: str | Path,
                                       graph_store: GraphStore, basin_ids: list[str]) -> Path:
    """
    Copy `source_db_path` (the submodule's own attributes.db for a run) to
    `target_db_path`, then add/overwrite the 7 violation_rate_R0..R6 columns for every
    basin in `basin_ids` in the copy. The source is never modified.

    Injection mechanism: the submodule's CamelsH5 dataset already looks up each
    sample's basin id and pulls static attributes directly from the sqlite database at
    `db_path`, z-score-normalizing whatever columns load_attributes returns (anything
    not in its INVALID_ATTR list). Adding these as ordinary extra columns means
    CamelsH5 includes and normalizes them automatically, with zero submodule changes.

    Zero-variance guard: if a rule has zero variance across all basins (e.g. no basin
    has any R6 violations yet), CamelsH5's z-score normalization would divide by zero
    for that column. A tiny deterministic jitter breaks ties in that case.
    """
    import shutil
    import sqlite3

    source_db_path = Path(source_db_path)
    target_db_path = Path(target_db_path)
    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_db_path, target_db_path)

    embeddings = build_embedding_matrix(graph_store, basin_ids)
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
    from `db_path` (post INVALID_ATTR filtering) -- use instead of a hardcoded count."""
    from hydrokg_adapters import load_camels_attributes
    return load_camels_attributes(db_path, basins).shape[1]


# ============================================================================
# Mechanism 3: graph-analogy correction (post-hoc, inference-time only)
# ============================================================================


class GraphAnalogyCorrector:
    """
    When a daily rule (R0-R3) fires for basin b at time t, HydroKG traverses to
    structurally similar basins (same aridity/land-cover class, low historical
    violation rate for that specific rule) and uses THEIR behavior, scaled to basin b's
    own flow regime, to produce a correction. Retrieval-based, not a physical clip and
    not a differentiable penalty -- applied once, at inference, after the LSTM has
    already produced its raw prediction.

    Scope: intended for daily, point-wise rules (R0-R3). R4/R5/R6 are event/annual-scale
    by construction and are not corrected point-by-point here.

    Mechanism: each analog basin's prediction at a comparable time is expressed
    relative to that basin's OWN long-term median flow (a scale-free ratio), then that
    ratio is transferred to basin b's own long-term median flow -- avoiding naively
    comparing raw discharge across basins of very different size/climate.
    """

    def __init__(self, graph_store: GraphStore, basin_dataframes: dict[str, pd.DataFrame],
                 min_analog_median: float = 1e-3):
        self.graph = graph_store
        self.basin_dataframes = basin_dataframes
        self.min_analog_median = min_analog_median
        self._median_cache: dict[str, float] = {}

    def _basin_median_qsim(self, basin_id: str) -> float:
        if basin_id not in self._median_cache:
            df = self.basin_dataframes.get(basin_id)
            median = float(df["qsim"].median()) if df is not None and not df.empty else np.nan
            self._median_cache[basin_id] = max(median, self.min_analog_median) if not np.isnan(median) else np.nan
        return self._median_cache[basin_id]

    def correct(self, basin_id: str, timestamp, q_sim_raw: float, rule_id: str,
                aridity_class: Optional[str], landcover_class: Optional[str],
                top_k: int = 5) -> tuple[float, dict]:
        """Returns (corrected_value, info); info records which analogs were used and the
        correction ratio applied, for auditability."""
        analogs = self.graph.query_analog_basins(basin_id, rule_id, aridity_class, landcover_class, top_k=top_k)
        if not analogs:
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_analogs_fallback", "analogs": []}

        ts = pd.Timestamp(timestamp)
        ratios, used = [], []
        for analog_id, score in analogs:
            analog_df = self.basin_dataframes.get(analog_id)
            if analog_df is None or ts not in analog_df.index:
                continue
            analog_median = self._basin_median_qsim(analog_id)
            if np.isnan(analog_median) or analog_median <= 0:
                continue
            ratios.append(analog_df.loc[ts, "qsim"] / analog_median)
            used.append(analog_id)

        if not ratios:
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_temporal_overlap_fallback", "analogs": [a for a, _ in analogs]}

        target_median = self._basin_median_qsim(basin_id)
        if np.isnan(target_median):
            fallback = max(q_sim_raw, 0.0) if rule_id in ("R0", "R2") else q_sim_raw
            return fallback, {"method": "no_target_median_fallback", "analogs": used}

        analog_ratio = float(np.median(ratios))
        corrected = target_median * analog_ratio
        if rule_id in ("R0", "R2"):
            corrected = max(corrected, 0.0)  # never let the correction itself create a new impossibility

        return corrected, {
            "method": "graph_analogy", "analogs": used, "analog_ratio": analog_ratio,
            "target_median": target_median, "raw_value": q_sim_raw,
        }


# ============================================================================
# Orchestration: the full training pipeline against the real submodule LSTM
# ============================================================================


class EnhancedTrainingPipeline:

    def __init__(self, graph_store: GraphStore, run_dir: str | Path, camels_root: str | Path,
                 work_dir: Optional[str | Path] = None):
        """
        Parameters
        ----------
        graph_store : should already have the BASELINE (traditional-LSTM) audit's
            violations written to it.
        work_dir : where to write the attributes.db copy, train_data.h5 (if missing),
            and enhanced outputs. Defaults to `<run_dir>/hydrokg_enhanced/`.
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

        from hydrokg_adapters import create_h5_dataset
        train_start = pd.to_datetime(self.run_cfg["train_start"], format="%d%m%Y")
        train_end = pd.to_datetime(self.run_cfg["train_end"], format="%d%m%Y")
        tqdm.write(
            f"[fine_tune] train_data.h5 not found at {original_h5} (gitignored preprocessing "
            f"artifact) -- rebuilding it once from CAMELS forcing/discharge under {rebuilt_h5}."
        )
        create_h5_dataset(
            camels_root=self.camels_root, out_file=rebuilt_h5, basins=basin_ids,
            train_start=train_start, train_end=train_end, seq_length=self.run_cfg["seq_length"],
        )
        return rebuilt_h5

    def fine_tune(self, basin_ids: list[str], n_epochs: int = 3, learning_rate: float = 5e-4,
                  device: str = "cpu") -> tuple[dict, Path]:
        """See module docstring for the full real-time loop description. Returns
        (state_dict, augmented_db_path)."""
        if self.run_cfg["no_static"]:
            raise ValueError(
                "This run was trained with no_static=True: there is no static-attribute "
                "channel to inject the violation-history embedding into. Re-train with "
                "concat_static=True, or use only curriculum reweighting and graph-analogy "
                "correction (which don't require it)."
            )

        _ensure_submodule_on_path()
        import torch
        from torch.utils.data import WeightedRandomSampler

        from data.datasets import CamelsH5  # noqa: E402 (submodule import)
        from data.datautils import rescale_features  # noqa: E402
        from src.main import Model  # noqa: E402 (submodule import)
        from Scripts.nseloss import NSELoss  # noqa: E402

        from hydrokg_rules import DAILY_RULES, build_all_rules
        daily_rules = {rid: rule for rid, rule in build_all_rules().items() if rid in DAILY_RULES}

        device_t = torch.device(device)
        tqdm.write("[fine_tune] Step B: locating/rebuilding train_data.h5")
        train_h5 = self._ensure_train_h5(basin_ids)

        tqdm.write("[fine_tune] Step A: preparing embedding-augmented attributes.db (epoch 1)")
        augmented_db = self._augmented_db_path(basin_ids)
        n_static = n_static_features(augmented_db, basin_ids)
        input_size_dyn = 5 + n_static

        model = Model(
            input_size_dyn=input_size_dyn, hidden_size=self.run_cfg["hidden_size"],
            initial_forget_bias=self.run_cfg.get("initial_forget_gate_bias", 5),
            dropout=self.run_cfg["dropout"], concat_static=True, no_static=False,
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
        tqdm.write(f"[fine_tune]   warm-started {len(compatible)}/{len(checkpoint_state)} parameters; "
                   f"random-init: {skipped or 'none'}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = NSELoss()
        batch_size = self.run_cfg["batch_size"]

        tqdm.write("[fine_tune] Step D: loading training data (once for the whole run)")
        ds = CamelsH5(h5_file=train_h5, basins=basin_ids, db_path=str(augmented_db),
                      concat_static=True, cache=True, no_static=False)

        tqdm.write(f"[fine_tune] Step E: training for {n_epochs} epoch(s), graph refreshed between each")
        model.train()
        for epoch in range(1, n_epochs + 1):
            if epoch > 1:
                tqdm.write(f"[fine_tune] Refreshing attributes.db + curriculum weights from "
                           f"epoch {epoch - 1}'s online detections")
                augmented_db = self._augmented_db_path(basin_ids)
                ds.db_path = str(augmented_db)
                ds._load_attributes()  # only refreshes the small attribute table, not x/y arrays

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

                with torch.no_grad():
                    q_sim_phys = rescale_features(predictions.detach().cpu().numpy(), "output").flatten()
                    q_obs_phys = rescale_features(y.detach().cpu().numpy(), "output").flatten()
                violations = self._detect_online_violations(batch_basins, q_obs_phys, q_sim_phys, daily_rules)
                if violations:
                    self.graph.write_violations(violations)
                    epoch_violations += len(violations)

                pbar.set_postfix(mean_loss=f"{running_loss / (b + 1):.4f}", violations=epoch_violations)

            tqdm.write(f"[fine_tune]   epoch {epoch}/{n_epochs} done: mean loss={running_loss / max(n_batches, 1):.4f}, "
                       f"{epoch_violations} online R0-R3 violations detected this epoch")

        state_dict_path = self.work_dir / "enhanced_model_state_dict.pt"
        torch.save(model.state_dict(), state_dict_path)
        return model.state_dict(), augmented_db

    @staticmethod
    def _detect_online_violations(batch_basins, q_obs_phys, q_sim_phys, daily_rules) -> list:
        """Group one training batch's (basin, qobs, qsim) triples by basin and run R0-R3
        against each basin's rows. Dates are synthetic PLACEHOLDER LABELS (a plain daily
        range, reset per basin per batch) purely to satisfy Rule.evaluate()'s pandas-
        datetime-indexed interface -- R0-R3 don't use calendar context, so these labels'
        specific values never affect anything; the qobs/qsim VALUES are always real,
        rescaled model output."""
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

    def generate_predictions(self, state_dict: dict, augmented_db_path: str | Path,
                              basin_ids: list[str], device: str = "cpu") -> dict[str, pd.DataFrame]:
        """Runs the fine-tuned model over the submodule's validation period
        (GLOBAL_SETTINGS val_start/val_end), mirroring its evaluate_basin() logic but
        against the fine-tuned weights and embedding-augmented attributes.db. Returns
        {basin_id: DataFrame(qobs, qsim)}."""
        _ensure_submodule_on_path()
        import torch
        from torch.utils.data import DataLoader

        from data.datasets import CamelsH5, CamelsTXT  # noqa: E402
        from data.datautils import rescale_features  # noqa: E402
        from src.main import Model, GLOBAL_SETTINGS  # noqa: E402

        device_t = torch.device(device)
        n_static = n_static_features(augmented_db_path, basin_ids)
        input_size_dyn = 5 + n_static

        model = Model(input_size_dyn=input_size_dyn, hidden_size=self.run_cfg["hidden_size"],
                       dropout=self.run_cfg["dropout"], concat_static=True, no_static=False).to(device_t)
        model.load_state_dict(state_dict)
        model.eval()

        train_h5 = self._ensure_train_h5(basin_ids)
        ds_train = CamelsH5(h5_file=train_h5, db_path=str(augmented_db_path), basins=basin_ids, concat_static=True)
        means = ds_train.get_attribute_means()
        stds = ds_train.get_attribute_stds()

        date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
        results: dict[str, pd.DataFrame] = {}
        skipped_basins = []

        for basin in tqdm(basin_ids, desc="generating predictions", unit="basin"):
            try:
                ds_test = CamelsTXT(
                    camels_root=self.camels_root, basin=basin,
                    dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]], is_train=False,
                    seq_length=self.run_cfg["seq_length"], with_attributes=True,
                    attribute_means=means, attribute_stds=stds, concat_static=True,
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
            results[basin] = pd.DataFrame({"qobs": obs_np[:n], "qsim": preds_np[:n]}, index=date_range[:n])

        if skipped_basins:
            tqdm.write(f"[generate_predictions] skipped {len(skipped_basins)} basin(s), e.g.: {skipped_basins[:3]}")
        return results

    def apply_analogy_correction(self, raw_predictions: dict[str, pd.DataFrame],
                                  violation_by_basin: dict[str, list[tuple[str, str]]],
                                  stratification: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Post-processing: for every (basin, timestamp) where a daily rule (R0-R3) was
        flagged, replace raw q_sim with the graph-analogy-corrected value."""
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
                    continue
                ts = pd.Timestamp(ts_iso)
                if ts not in corrected[basin_id].index:
                    continue
                raw_val = corrected[basin_id].loc[ts, "qsim"]
                new_val, _info = corrector.correct(basin_id, ts, raw_val, rule_id, aridity_class, landcover_class)
                corrected[basin_id].loc[ts, "qsim"] = new_val
                n_corrected += 1
            pbar.set_postfix(corrected=n_corrected, of=n_flagged)

        return corrected

    def save_predictions_pickle(self, predictions: dict[str, pd.DataFrame],
                                 filename: str = "enhanced_predictions.p") -> Path:
        """Save in the same {basin_id: DataFrame(qobs, qsim)} shape as the submodule's
        own evaluate() output."""
        out_path = self.work_dir / filename
        with open(out_path, "wb") as fp:
            pickle.dump(predictions, fp)
        tqdm.write(f"[save_predictions_pickle] saved enhanced predictions to {out_path}")
        return out_path
