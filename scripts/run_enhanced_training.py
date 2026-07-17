"""
Enhancement CLI -- runs the full graph-guided enhancement pipeline against a real
trained LSTM: baseline audit -> curriculum-reweighted + violation-embedding-augmented
real-time fine-tuning -> regenerate predictions -> re-audit -> graph-analogy correction
-> final audit. Produces two OfflineAuditor.audit_all() DataFrames (baseline_results,
enhanced_results) saved as CSVs, ready for hydrokg_evaluation.compute_deltas and the
skill-trust/enhancement figure panels. Real data only; no demo/synthetic mode.

    python scripts/run_enhanced_training.py \
        --run_dir external/HydroAuditToolFrameowrk/runs/<run> \
        --camels_root /path/to/CAMELS_US \
        --predictions_pickle external/HydroAuditToolFrameowrk/runs/<run>/lstm_seed<seed>.p \
        --n_epochs 3 --n_workers 8 --output_prefix results/hydrokg_run1

Requires: pip install -r requirements-torch.txt

CHECKPOINTING: every step's output is written to disk as soon as it completes (baseline
results after step 1, the fine-tuned checkpoint after step 2, RAW predictions
immediately after step 3 -- not only after correction, which was the gap that lost an
entire ~16-hour run to a walltime kill during step 4). Pass --resume to skip any step
whose output file already exists rather than recomputing it -- a killed job can be
resubmitted and pick up from wherever it actually got to, instead of restarting at step 1.

PARALLELIZATION: steps 3 (per-basin prediction generation) and 4a (per-basin violation
scanning) are basin-independent and now run concurrently across --n_workers workers
within this single job -- no SLURM job array needed. Step 2 (fine-tuning) stays
sequential (each epoch depends on the previous epoch's weights and updated graph state)
and is the one part of the pipeline this change does NOT speed up.

Output is intentionally quiet: five clear step banners plus tqdm progress bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Flat repo layout: no installed package -- add the sibling src/ directory to
# sys.path so `import hydrokg_*` resolves regardless of the current working
# directory this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import logging
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from hydrokg_audit import OfflineAuditor
from hydrokg_graph import build_graph_store

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("hydrokg").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def step(n: int, total: int, message: str) -> None:
    tqdm.write(f"\n[Step {n}/{total}] {message}")


def _scan_one_basin(args):
    """Top-level (picklable) worker for ProcessPoolExecutor: evaluate all 7 rules
    against one basin's DataFrame. Pure CPU-bound pandas work, no shared state, no
    graph writes -- exactly why this is safe and worthwhile to run in separate
    processes rather than threads (bypasses the GIL for real parallel CPU execution)."""
    basin_id, df, aridity_class, landcover_class = args
    from hydrokg_rules import build_all_rules
    violations = []
    for rule_id, rule in build_all_rules().items():
        for v in rule.evaluate(basin_id, df, aridity_class, landcover_class):
            violations.append((v.timestamp.isoformat(), rule_id))
    return basin_id, violations


def run_real(args):
    from hydrokg_adapters import load_predictions_pickle
    from hydrokg_data import load_basin_stratification, attach_precipitation
    from hydrokg_enhancement import EnhancedTrainingPipeline
    from hydrokg_evaluation import enhancement_summary

    baseline_csv = args.output_prefix + "_baseline_results.csv"
    finetune_ckpt = None  # resolved below once we know run_dir
    raw_pred_pickle = args.output_prefix + "_raw_enhanced_predictions.p"
    enhanced_pred_pickle = args.output_prefix + "_enhanced_predictions.p"
    enhanced_csv = args.output_prefix + "_enhanced_results.csv"

    graph = build_graph_store("memory") if args.graph_backend == "memory" else build_graph_store(
        "neo4j", uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password
    )

    # --- 1. Baseline audit (seeds curriculum weights + violation embeddings) ---
    step(1, 5, "Baseline audit of the traditional LSTM's predictions")
    if args.resume and Path(baseline_csv).exists():
        tqdm.write(f"  --resume: found {baseline_csv}, loading instead of re-auditing")
        baseline_results = pd.read_csv(baseline_csv, dtype={"basin_id": str})
        basin_ids = baseline_results["basin_id"].astype(str).tolist()
        stratification = (
            load_basin_stratification(args.stratification_db, basin_ids)
            if args.stratification_db else pd.DataFrame(index=basin_ids)
        )
        # NOTE: on resume, the graph does NOT contain the baseline audit's violations
        # (they were never persisted beyond the CSV summary) -- curriculum
        # reweighting/embeddings for a resumed fine-tune step will start from an empty
        # graph rather than the original baseline state. This only matters if you are
        # resuming specifically INTO step 2; resuming at step 3+ is unaffected since
        # fine_tune() has already run and its own graph writes happened in that process.
    else:
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
        baseline_results.to_csv(baseline_csv, index=False)
        basin_ids = list(baseline_basins.keys())
    tqdm.write(f"  {len(baseline_results)} basins, mean violation_burden="
               f"{baseline_results['violation_burden'].mean():.4f}")

    # --- 2. Fine-tune with curriculum reweighting + violation embeddings (real-time loop) ---
    step(2, 5, "Fine-tuning with curriculum reweighting + violation embeddings")
    pipeline = EnhancedTrainingPipeline(graph, run_dir=args.run_dir, camels_root=args.camels_root)
    finetune_ckpt = pipeline.work_dir / "enhanced_model_state_dict.pt"

    if args.resume and finetune_ckpt.exists() and Path(raw_pred_pickle).exists():
        tqdm.write(f"  --resume: found {finetune_ckpt.name} and raw predictions, skipping fine-tuning entirely")
        state_dict = augmented_db = None  # not needed; step 3 also skipped below
    else:
        state_dict, augmented_db = pipeline.fine_tune(
            basin_ids, n_epochs=args.n_epochs, learning_rate=args.learning_rate, device=args.device
        )

    # --- 3. Regenerate predictions from the fine-tuned model (parallel across basins) ---
    step(3, 5, "Generating predictions from the fine-tuned model")
    if args.resume and Path(raw_pred_pickle).exists():
        tqdm.write(f"  --resume: found {raw_pred_pickle}, loading instead of regenerating")
        with open(raw_pred_pickle, "rb") as fp:
            raw_enhanced = pickle.load(fp)
    else:
        raw_enhanced = pipeline.generate_predictions(
            state_dict, augmented_db, basin_ids, device=args.device, n_workers=args.n_workers
        )
        raw_enhanced = {
            b: attach_precipitation(df, args.camels_root, b)
            for b, df in tqdm(raw_enhanced.items(), desc="attaching precipitation", unit="basin")
        }
        # Checkpoint immediately -- this is the exact output a walltime kill during
        # step 4 previously lost entirely, forcing a full restart from step 1.
        with open(raw_pred_pickle, "wb") as fp:
            pickle.dump(raw_enhanced, fp)
        tqdm.write(f"  checkpointed raw predictions to {raw_pred_pickle}")

    # --- 4. Audit the fine-tuned model's raw output (parallel), then graph-analogy correct ---
    step(4, 5, "Auditing fine-tuned output and applying graph-analogy correction")
    if args.resume and Path(enhanced_pred_pickle).exists():
        tqdm.write(f"  --resume: found {enhanced_pred_pickle}, skipping correction entirely")
        with open(enhanced_pred_pickle, "rb") as fp:
            corrected = pickle.load(fp)
    else:
        strat_lookup = {
            b: (stratification.loc[b].get("aridity_class") if b in stratification.index else None,
                stratification.loc[b].get("landcover_class") if b in stratification.index else None)
            for b in raw_enhanced
        }

        violation_by_basin: dict[str, list[tuple[str, str]]] = {b: [] for b in raw_enhanced}
        scan_args = [(b, df, *strat_lookup[b]) for b, df in raw_enhanced.items()]

        if args.n_workers <= 1:
            for a in tqdm(scan_args, desc="scanning for remaining violations", unit="basin"):
                basin_id, violations = _scan_one_basin(a)
                violation_by_basin[basin_id] = violations
        else:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futures = [pool.submit(_scan_one_basin, a) for a in scan_args]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"scanning for remaining violations ({args.n_workers} workers)",
                                   unit="basin"):
                    basin_id, violations = future.result()
                    violation_by_basin[basin_id] = violations

        corrected = pipeline.apply_analogy_correction(raw_enhanced, violation_by_basin, stratification)
        pipeline.save_predictions_pickle(corrected, filename=Path(enhanced_pred_pickle).name)

    # --- 5. Final audit of the corrected enhanced predictions ---
    step(5, 5, "Final audit of the corrected enhanced predictions")
    final_graph = build_graph_store("memory")
    final_auditor = OfflineAuditor(final_graph)
    enhanced_results = final_auditor.audit_all(corrected, stratification)
    enhanced_results.to_csv(enhanced_csv, index=False)

    summary = enhancement_summary(baseline_results, enhanced_results)
    tqdm.write("\n=== Enhancement summary ===")
    for key, value in summary.items():
        if key != "deltas":
            tqdm.write(f"  {key}: {value}")

    graph.close()
    tqdm.write(f"\nDone. Saved: {baseline_csv}, {enhanced_csv}, {enhanced_pred_pickle}")


def main():
    parser = argparse.ArgumentParser(description="HydroKG graph-guided enhancement pipeline")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--camels_root", type=str, required=True)
    parser.add_argument("--predictions_pickle", type=str, required=True)
    parser.add_argument("--stratification_db", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallelize per-basin prediction generation and violation scanning "
                             "(steps 3 and 4a) across this many workers. Match to your SLURM -c value.")
    parser.add_argument("--graph_backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="hydrokg_enhancement")
    parser.add_argument("--resume", action="store_true",
                        help="Skip any step whose output file already exists instead of "
                             "recomputing it -- resume a killed/timed-out job.")
    args = parser.parse_args()

    run_real(args)


if __name__ == "__main__":
    main()