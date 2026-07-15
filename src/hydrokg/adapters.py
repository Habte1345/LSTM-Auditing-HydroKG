"""
Adapter around the external/HydroAuditToolFrameowrk submodule.

The submodule is kept completely untouched -- it has hardcoded local paths in its
committed cfg.json and no packaging (setup.py), so importing its `data`/`Scripts`
modules requires putting its repo root on sys.path, exactly as its own src/main.py does
internally. This module is the ONLY place that sys.path hack lives.

Everything here is config-driven (camels_root, run_dir passed as arguments) -- no
hardcoded paths.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2] / "external" / "HydroAuditToolFrameowrk"


def _ensure_submodule_on_path() -> Path:
    if not _SUBMODULE_ROOT.exists():
        raise FileNotFoundError(
            f"Submodule not found at {_SUBMODULE_ROOT}. Run "
            "`git submodule update --init --recursive` from the repo root."
        )
    root_str = str(_SUBMODULE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return _SUBMODULE_ROOT


def load_predictions_pickle(pickle_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the {basin_id: DataFrame(qobs, qsim)} dict produced by the submodule's
    src/main.py `evaluate` mode."""
    with open(pickle_path, "rb") as fp:
        results: dict[str, pd.DataFrame] = pickle.load(fp)
    return results


def load_run_config(run_dir: str | Path) -> dict:
    """Read the submodule's own cfg.json for a completed run."""
    with open(Path(run_dir) / "cfg.json", "r") as fp:
        return json.load(fp)


def load_forcing_for_basin(camels_root: str | Path, basin_id: str) -> tuple[pd.DataFrame, int]:
    """Delegates to the submodule's data.datautils.load_forcing, unmodified."""
    _ensure_submodule_on_path()
    from data.datautils import load_forcing  # noqa: E402
    return load_forcing(Path(camels_root), basin_id)


def load_discharge_for_basin(camels_root: str | Path, basin_id: str, area: int) -> pd.Series:
    """Delegates to the submodule's data.datautils.load_discharge, unmodified."""
    _ensure_submodule_on_path()
    from data.datautils import load_discharge  # noqa: E402
    return load_discharge(Path(camels_root), basin_id, area)


def load_camels_attributes(db_path: str | Path, basins: list[str],
                            keep_features: Optional[list[str]] = None) -> pd.DataFrame:
    """Delegates to the submodule's data.datautils.load_attributes, unmodified. Pass
    keep_features to retrieve stratification attributes (e.g. 'aridity',
    'dom_land_cover') that the submodule's own LSTM training excludes."""
    _ensure_submodule_on_path()
    from data.datautils import load_attributes  # noqa: E402
    return load_attributes(str(db_path), basins, keep_features=keep_features)


def get_basin_list() -> list[str]:
    """Delegates to the submodule's Scripts.utils.get_basin_list, unmodified."""
    _ensure_submodule_on_path()
    from Scripts.utils import get_basin_list  # noqa: E402
    return get_basin_list()


def create_h5_dataset(camels_root: str | Path, out_file: str | Path, basins: list[str],
                       train_start, train_end, seq_length: int) -> Path:
    """Delegates to the submodule's Scripts.utils.create_h5_files, unmodified.

    train_data.h5 is a large preprocessing artifact the submodule's own training run
    builds once and is correctly gitignored -- it will NOT exist in a fresh submodule
    clone even though a run's cfg.json/model weights do. Rebuilds it on demand (raises
    FileExistsError if already there, matching the submodule's own behavior)."""
    _ensure_submodule_on_path()
    from Scripts.utils import create_h5_files  # noqa: E402

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    create_h5_files(
        camels_root=Path(camels_root), out_file=out_file, basins=basins,
        dates=[pd.Timestamp(train_start), pd.Timestamp(train_end)],
        with_basin_str=True, seq_length=seq_length,
    )
    return out_file


def run_submodule_cli(mode: str, extra_args: Optional[list[str]] = None) -> subprocess.CompletedProcess:
    """Invoke the submodule's src/main.py as a subprocess (train/evaluate/create_splits)."""
    submodule_root = _ensure_submodule_on_path()
    cmd = [sys.executable, str(submodule_root / "src" / "main.py"), mode] + (extra_args or [])
    return subprocess.run(cmd, cwd=str(submodule_root), capture_output=True, text=True)
