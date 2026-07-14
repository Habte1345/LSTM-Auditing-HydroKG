"""
Precipitation loading for R5/R6.

Precipitation is read from CAMELS forcing via the untouched submodule
(hydrokg.adapters.load_forcing_for_basin), already in mm/day at the same basin-area
normalization convention the submodule's own load_discharge uses for Q (so P and Q are
directly comparable without an additional area-normalization step here).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hydrokg.adapters.lstm_adapter import load_forcing_for_basin

# CAMELS Maurer/NLDAS forcing files use varying header capitalization across releases;
# match case-insensitively on the substring rather than hardcoding one exact column name.
_PRECIP_COLUMN_HINTS = ("prcp", "precip")


def _find_precip_column(forcing_df: pd.DataFrame) -> str:
    for col in forcing_df.columns:
        low = col.lower()
        if any(hint in low for hint in _PRECIP_COLUMN_HINTS):
            return col
    raise KeyError(
        f"No precipitation column found in forcing data (columns: {list(forcing_df.columns)}). "
        "CAMELS forcing files should include a PRCP column; verify camels_root points at the "
        "correct forcing product (nldas_extended)."
    )


def load_precipitation_series(camels_root: str | Path, basin_id: str) -> pd.Series:
    """Return a daily precipitation series (mm/day) for one basin, indexed by date."""
    forcing_df, _area = load_forcing_for_basin(camels_root, basin_id)
    precip_col = _find_precip_column(forcing_df)
    series = forcing_df[precip_col].copy()
    series.name = "p"
    return series


def attach_precipitation(df: pd.DataFrame, camels_root: str | Path, basin_id: str) -> pd.DataFrame:
    """Left-join precipitation onto a basin's qobs/qsim DataFrame (same date index)."""
    p_series = load_precipitation_series(camels_root, basin_id)
    out = df.join(p_series, how="left")
    return out
