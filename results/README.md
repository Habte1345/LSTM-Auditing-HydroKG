# results/

Default output location for CLI runs (baseline/enhanced audit CSVs, enhanced
predictions pickle). Point `--output_prefix results/<name>` at either CLI to land
outputs here, e.g.:

```
python -m hydrokg.cli.run_enhanced_training ... --output_prefix results/hydrokg_run1
```

produces `results/hydrokg_run1_baseline_results.csv`,
`results/hydrokg_run1_enhanced_results.csv`, and
`results/hydrokg_run1_enhanced_predictions.p`.

This directory is gitignored by default (run outputs are regenerable, not source) --
remove the ignore rule in `.gitignore` if you want to version specific result CSVs.
