"""
Enhancement CLI.

Demo mode (synthetic data, no CAMELS/torch/submodule checkpoint needed):
    python -m hydrokg.cli.run_enhanced_training --demo

Real mode: runs the FULL pipeline against your actual trained LSTM --
baseline audit -> curriculum-reweighted + violation-embedding-augmented fine-tuning ->
regenerate predictions -> re-audit -> graph-analogy correction -> final audit. Produces
two OfflineAuditor.audit_all() DataFrames (baseline_results, enhanced_results) saved as
CSVs, ready for hydrokg.evaluation.enhancement_metrics.compute_deltas and the (e)/(f)
panels of the skill-trust figure.

    python -m hydrokg.cli.run_enhanced_training \
        --run_dir external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666 \
        --camels_root "F:/Data/CAMEL_SI/CAMELS_US/" \
        --predictions_pickle external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666/lstm_seed658.p \
        --n_epochs 3

Requires the `torch` extra: pip install -e ".[torch]"

Output is intentionally quiet: five clear step banners plus tqdm progress bars for the
long-running loops (fine-tuning batches, per-basin prediction generation, per-basin
correction), rather than one log line per basin/epoch/warning.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd
from tqdm import tqdm

from hydrokg.audit.offline_auditor import OfflineAuditor
from hydrokg.data.synthetic import make_synthetic_basin
from hydrokg.enhancement.curriculum import ViolationCurriculumSampler
from hydrokg.enhancement.graph_analogy_correction import GraphAnalogyCorrector
from hydrokg.enhancement.violation_embeddings import build_embedding_matrix
from hydrokg.graph.factory import build_graph_store

# Only our own step banners and warnings should reach the console; third-party library
# INFO/WARNING noise (pandas, torch, h5py) is suppressed here rather than left to flood
# the terminal alongside the pipeline's actual progress.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("hydrokg").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def step(n: int, total: int, message: str) -> None:
    tqdm.write(f"\n[Step {n}/{total}] {message}")


def run_demo():
    graph = build_graph_store("memory")

    basins = {
        "DEMO0001": make_synthetic_basin("DEMO0001", seed=1),
        "DEMO0002": make_synthetic_basin("DEMO0002", seed=2, inject_negative_flow=True),
        "DEMO0003": make_synthetic_basin("DEMO0003", seed=3, inject_negative_flow=True),
        "DEMO0004": make_synthetic_basin("DEMO0004", seed=4),
    }
    strat = pd.DataFrame({
        "aridity_class": ["humid", "humid", "humid", "humid"],
        "landcover_class": ["forest", "forest", "forest", "forest"],
    }, index=list(basins.keys()))

    auditor = OfflineAuditor(graph)
    results = auditor.audit_all(basins, strat)
    tqdm.write(f"Baseline audit:\n{results[['basin_id', 'kge', 'violation_burden']]}")

    sampler = ViolationCurriculumSampler(graph)
    tqdm.write(f"Curriculum sampling weights: {sampler.basin_weights(list(basins.keys()))}")
    tqdm.write(f"Violation-history embeddings:\n{build_embedding_matrix(graph, list(basins.keys()))}")

    corrector = GraphAnalogyCorrector(graph, basins)
    negative_rows = basins["DEMO0002"][basins["DEMO0002"]["qsim"] < 0]
    for ts, row in negative_rows.head(3).iterrows():
        corrected_val, info = corrector.correct("DEMO0002", ts, row["qsim"], "R0", "humid", "forest")
        tqdm.write(f"Corrected DEMO0002 @ {ts.date()}: raw={row['qsim']:.3f} -> "
                    f"corrected={corrected_val:.3f} ({info['method']})")
    graph.close()


def run_real(args):
    from hydrokg.adapters.lstm_adapter import load_predictions_pickle
    from hydrokg.data.basin_attributes import load_basin_stratification
    from hydrokg.data.forcing_loader import attach_precipitation
    from hydrokg.enhancement.enhanced_training import EnhancedTrainingPipeline
    from hydrokg.evaluation.enhancement_metrics import enhancement_summary

    graph = build_graph_store("memory") if args.graph_backend == "memory" else build_graph_store(
        "neo4j", uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password
    )

    # --- 1. Baseline audit (seeds curriculum weights + violation embeddings) ---
    step(1, 5, "Baseline audit of the traditional LSTM's predictions")
    baseline_raw = load_predictions_pickle(args.predictions_pickle)
    baseline_basins = {
        b: attach_precipitation(df, args.camels_root, b)
        for b, df in tqdm(baseline_raw.items(), desc="attaching precipitation", unit="basin")
    }
    stratification = (
        load_basin_stratification(args.stratification_db, list(baseline_basins.keys()))
        if args.stratification_db else pd.DataFrame(index=list(baseline_basins.keys()))
    )
    auditor = OfflineAuditor(graph)
    baseline_results = auditor.audit_all(baseline_basins, stratification)
    baseline_results.to_csv(args.output_prefix + "_baseline_results.csv", index=False)
    tqdm.write(f"  {len(baseline_results)} basins audited, "
               f"mean violation_burden={baseline_results['violation_burden'].mean():.4f}")

    # --- 2. Fine-tune with curriculum reweighting + violation embeddings ---
    step(2, 5, "Fine-tuning with curriculum reweighting + violation embeddings")
    pipeline = EnhancedTrainingPipeline(graph, run_dir=args.run_dir, camels_root=args.camels_root)
    basin_ids = list(baseline_basins.keys())
    state_dict, augmented_db = pipeline.fine_tune(
        basin_ids, n_epochs=args.n_epochs, learning_rate=args.learning_rate, device=args.device
    )

    # --- 3. Regenerate predictions from the fine-tuned model ---
    step(3, 5, "Generating predictions from the fine-tuned model")
    raw_enhanced = pipeline.generate_predictions(state_dict, augmented_db, basin_ids, device=args.device)
    raw_enhanced = {
        b: attach_precipitation(df, args.camels_root, b)
        for b, df in tqdm(raw_enhanced.items(), desc="attaching precipitation", unit="basin")
    }

    # --- 4. Audit the fine-tuned model's raw output, then apply graph-analogy correction ---
    step(4, 5, "Auditing fine-tuned output and applying graph-analogy correction")
    post_finetune_graph = build_graph_store("memory")  # separate from the baseline graph
    post_auditor = OfflineAuditor(post_finetune_graph)
    for basin_id in raw_enhanced:
        arid = stratification.loc[basin_id].get("aridity_class") if basin_id in stratification.index else None
        land = stratification.loc[basin_id].get("landcover_class") if basin_id in stratification.index else None
        post_finetune_graph.register_catchment(basin_id, arid, land)

    violation_by_basin: dict[str, list[tuple[str, str]]] = {b: [] for b in raw_enhanced}
    for basin_id, df in tqdm(raw_enhanced.items(), desc="scanning for remaining violations", unit="basin"):
        arid = stratification.loc[basin_id].get("aridity_class") if basin_id in stratification.index else None
        land = stratification.loc[basin_id].get("landcover_class") if basin_id in stratification.index else None
        for rule_id, rule in post_auditor.rules.items():
            for v in rule.evaluate(basin_id, df, arid, land):
                violation_by_basin[basin_id].append((v.timestamp.isoformat(), rule_id))

    corrected = pipeline.apply_analogy_correction(raw_enhanced, violation_by_basin, stratification)
    pipeline.save_predictions_pickle(corrected, filename=args.output_prefix + "_enhanced_predictions.p")

    # --- 5. Final audit of the corrected enhanced predictions ---
    step(5, 5, "Final audit of the corrected enhanced predictions")
    final_graph = build_graph_store("memory")
    final_auditor = OfflineAuditor(final_graph)
    enhanced_results = final_auditor.audit_all(corrected, stratification)
    enhanced_results.to_csv(args.output_prefix + "_enhanced_results.csv", index=False)

    summary = enhancement_summary(baseline_results, enhanced_results)
    tqdm.write("\n=== Enhancement summary ===")
    for key, value in summary.items():
        if key != "deltas":
            tqdm.write(f"  {key}: {value}")

    graph.close()
    tqdm.write(
        f"\nDone. Saved:\n"
        f"  {args.output_prefix}_baseline_results.csv\n"
        f"  {args.output_prefix}_enhanced_results.csv\n"
        f"  {args.output_prefix}_enhanced_predictions.p\n"
        f"Load the two CSVs and pass to hydrokg.evaluation.enhancement_metrics.compute_deltas() "
        f"for the (e)/(f) figure panels."
    )


def main():
    parser = argparse.ArgumentParser(description="HydroKG graph-guided enhancement pipeline")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Submodule run directory (contains cfg.json, model_epoch5.pt, attributes.db)")
    parser.add_argument("--camels_root", type=str, default=None)
    parser.add_argument("--predictions_pickle", type=str, default=None,
                        help="Traditional (baseline) LSTM predictions pickle")
    parser.add_argument("--stratification_db", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--graph_backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="hydrokg_enhancement")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    missing = [name for name in ("run_dir", "camels_root", "predictions_pickle")
               if getattr(args, name) is None]
    if missing:
        parser.error(f"Real mode requires: {', '.join('--' + m for m in missing)}. "
                     f"Use --demo to try the mechanisms against synthetic data first.")

    run_real(args)


if __name__ == "__main__":
    main()
