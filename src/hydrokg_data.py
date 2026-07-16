"""
Data loading for R5/R6 and stratification: precipitation from CAMELS forcing,
ET as a long-term water-balance residual (for reporting only, not fed into R5/R6
directly -- see below), and aridity/land-cover stratification classes. Merged from
3 files; the synthetic-data generator that used to live alongside these has been
removed entirely -- this module now only touches real CAMELS data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hydrokg_adapters import load_camels_attributes, load_forcing_for_basin

# ============================================================================
# Precipitation loading (for R5/R6)
# ============================================================================

_PRECIP_COLUMN_HINTS = ("prcp", "precip")


def _find_precip_column(forcing_df: pd.DataFrame) -> str:
    for col in forcing_df.columns:
        if any(hint in col.lower() for hint in _PRECIP_COLUMN_HINTS):
            return col
    raise KeyError(
        f"No precipitation column found in forcing data (columns: {list(forcing_df.columns)}). "
        "Verify camels_root points at the correct forcing product (nldas_extended)."
    )


def load_precipitation_series(camels_root: str | Path, basin_id: str) -> pd.Series:
    """Daily precipitation series (mm/day) for one basin, indexed by date."""
    forcing_df, _area = load_forcing_for_basin(camels_root, basin_id)
    series = forcing_df[_find_precip_column(forcing_df)].copy()
    series.name = "p"
    return series


def attach_precipitation(df: pd.DataFrame, camels_root: str | Path, basin_id: str) -> pd.DataFrame:
    """Left-join precipitation onto a basin's qobs/qsim DataFrame (same date index)."""
    return df.join(load_precipitation_series(camels_root, basin_id), how="left")


# ============================================================================
# ET as a water-balance residual (long-term reporting only; R5/R6 use P and Q_sim
# directly, NOT this function -- see docs/RULES.md for why)
# ============================================================================


def long_term_et_residual(p: pd.Series, qobs: pd.Series, min_years: int = 5) -> float:
    """Long-term mean ET (mm/day) as the water-balance residual P - Q_obs, assuming
    dS/dt ~ 0 over multi-decade CAMELS records. Raises if fewer than min_years of
    overlapping data are available, since that assumption is unreliable over short periods."""
    aligned = pd.DataFrame({"p": p, "qobs": qobs}).dropna()
    n_years = aligned.index.to_series().dt.year.nunique() if not aligned.empty else 0
    if n_years < min_years:
        raise ValueError(f"Only {n_years} years of overlapping P/Qobs data; need >= {min_years}.")
    return float(aligned["p"].mean() - aligned["qobs"].mean())


def long_term_runoff_ratio(p: pd.Series, q: pd.Series) -> float:
    """mean(Q) / mean(P) -- for basin aridity/water-balance sanity checks in figures."""
    aligned = pd.DataFrame({"p": p, "q": q}).dropna()
    p_mean = aligned["p"].mean()
    if p_mean <= 0:
        return float("nan")
    return float(aligned["q"].mean() / p_mean)


# ============================================================================
# Aridity / land-cover stratification
# ============================================================================

STRATIFICATION_ATTRS = ["aridity", "dom_land_cover", "dom_land_cover_frac"]
ARIDITY_BINS = [-float("inf"), 0.5, 1.0, 2.0, float("inf")]
ARIDITY_LABELS = ["humid", "sub_humid", "semi_arid", "arid"]


def classify_aridity(aridity_index: float) -> str:
    if pd.isna(aridity_index):
        return "unknown"
    for i in range(len(ARIDITY_BINS) - 1):
        if ARIDITY_BINS[i] < aridity_index <= ARIDITY_BINS[i + 1]:
            return ARIDITY_LABELS[i]
    return "unknown"


def load_basin_stratification(db_path: str, basins: list[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by basin_id with columns aridity_class, landcover_class."""
    attrs = load_camels_attributes(db_path, basins, keep_features=STRATIFICATION_ATTRS)
    out = pd.DataFrame(index=attrs.index)
    out["aridity_index"] = attrs.get("aridity")
    out["aridity_class"] = out["aridity_index"].apply(classify_aridity)
    out["landcover_class"] = attrs.get("dom_land_cover")
    return out
