"""
Synthetic basin prediction generator.

Used by tests/ and `hydrokg-audit --demo` so the full pipeline can be exercised without
CAMELS data or a live LSTM run. Deliberately injects known, controlled violations of each
rule so that the audit output is checkable against ground truth (see tests/test_rules.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_basin(
    basin_id: str,
    n_years: int = 8,
    seed: int = 0,
    inject_negative_flow: bool = False,
    inject_extreme_ratio: bool = False,
    inject_zero_collapse: bool = False,
    inject_high_rel_error: bool = False,
    inject_peak_lag_days: int = 0,
    inject_mass_balance_violation: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns qobs, qsim, p."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-10-01", periods=365 * n_years, freq="D")
    n = len(dates)

    # seasonal, log-normal-ish synthetic discharge with a snowmelt-like spring pulse
    day_of_year = dates.dayofyear.values
    seasonal = 2.0 + 3.0 * np.clip(np.sin((day_of_year - 60) / 365 * 2 * np.pi), 0, None)
    noise = rng.lognormal(mean=0.0, sigma=0.3, size=n)
    qobs = np.maximum(seasonal * noise, 0.01)

    qsim = qobs * rng.normal(loc=1.0, scale=0.15, size=n)
    qsim = np.maximum(qsim, 0.0)

    p = qobs * rng.uniform(1.8, 2.6, size=n)  # precipitation exceeds runoff, physically sane

    df = pd.DataFrame({"qobs": qobs, "qsim": qsim, "p": p}, index=dates)

    if inject_negative_flow:
        idx = rng.choice(n, size=max(3, n // 200), replace=False)
        df.iloc[idx, df.columns.get_loc("qsim")] = -abs(df.iloc[idx]["qsim"].values) - 0.5

    if inject_extreme_ratio:
        idx = rng.choice(n, size=max(3, n // 150), replace=False)
        df.iloc[idx, df.columns.get_loc("qsim")] = df.iloc[idx]["qobs"].values * 8.0

    if inject_zero_collapse:
        idx = rng.choice(n, size=max(3, n // 150), replace=False)
        # only meaningful where qobs is well above the basin mean
        high_flow_idx = df.index[df["qobs"] > df["qobs"].mean()]
        idx = rng.choice(len(high_flow_idx), size=min(len(high_flow_idx), max(3, n // 150)), replace=False)
        df.loc[high_flow_idx[idx], "qsim"] = 0.0

    if inject_high_rel_error:
        idx = rng.choice(n, size=max(3, n // 150), replace=False)
        df.iloc[idx, df.columns.get_loc("qsim")] = df.iloc[idx]["qobs"].values * 3.5

    if inject_peak_lag_days:
        # shift the whole series' peak by rolling qsim
        df["qsim"] = np.roll(df["qsim"].values, inject_peak_lag_days)

    if inject_mass_balance_violation:
        # force simulated annual mean above precip mean for one water year
        wy_mask = (dates.year == dates.year[len(dates) // 2])
        df.loc[wy_mask, "qsim"] = df.loc[wy_mask, "p"] * 1.5

    df.attrs["basin_id"] = basin_id
    return df
