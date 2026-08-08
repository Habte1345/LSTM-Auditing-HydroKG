"""
Diagnoses whether R1/R3 violations are concentrated in low-flow periods (the
relative-error-blowup artifact) rather than reflecting genuine large prediction errors.

Usage: run against your real predictions pickle (traditional LSTM's output).
    python diagnose_lowflow_r1_r3.py --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run>/lstm_seed<seed>.p
"""
import sys
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
except NameError:
    repo_root = Path("/bighome/hdagne1/LSTM-Auditing-HydroKG")

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import pandas as pd

from hydrokg_adapters import load_predictions_pickle
from hydrokg_rules import ExtremeRatioRule, HighRelativeErrorRule

predictions_pickle = str(repo_root / "external" / "HydroAuditToolFrameowrk" / "runs" /
                          "run_0305_2015_seed658666" / "lstm_seed658.p")  # <-- edit if needed

basins = load_predictions_pickle(predictions_pickle)

r1 = ExtremeRatioRule()
r3 = HighRelativeErrorRule()

low_flow_r1, total_r1 = 0, 0
low_flow_r3, total_r3 = 0, 0

for basin_id, df in basins.items():
    basin_median_flow = df["qobs"].median()
    low_flow_threshold = 0.1 * basin_median_flow  # "low flow" = below 10% of this basin's own median

    for v in r1.evaluate(basin_id, df):
        total_r1 += 1
        if v.q_obs < low_flow_threshold:
            low_flow_r1 += 1

    for v in r3.evaluate(basin_id, df):
        total_r3 += 1
        if v.q_obs < low_flow_threshold:
            low_flow_r3 += 1

print(f"R1 (extreme ratio): {total_r1} total violations, "
      f"{low_flow_r1} ({100*low_flow_r1/max(total_r1,1):.1f}%) occur during low-flow days")
print(f"R3 (high relative error): {total_r3} total violations, "
      f"{low_flow_r3} ({100*low_flow_r3/max(total_r3,1):.1f}%) occur during low-flow days")
print()
print("If these percentages are high (e.g. >70%), R1/R3's dominance is very likely an")
print("artifact of dividing by near-zero observed flow, not genuine large-magnitude errors.")