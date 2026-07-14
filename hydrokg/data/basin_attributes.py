"""
Basin-level stratification classes (AridityClass, LandCoverClass in the ontology).

CAMELS' own 'aridity' attribute (PET/P, Budyko dryness index) and 'dom_land_cover' are
excluded from the LSTM's own static input features by the submodule's INVALID_ATTR list
(they're used for stratification/analysis in the original Kratzert et al. study design,
not as model inputs) -- HydroKG reads them directly via
hydrokg.adapters.load_camels_attributes(..., keep_features=[...]), bypassing that
exclusion, since here they're needed for exactly that stratification purpose, not as
model inputs.
"""

from __future__ import annotations

import pandas as pd

from hydrokg.adapters.lstm_adapter import load_camels_attributes

STRATIFICATION_ATTRS = ["aridity", "dom_land_cover", "dom_land_cover_frac"]

# Standard Budyko-style aridity index (PET/P) bins.
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
