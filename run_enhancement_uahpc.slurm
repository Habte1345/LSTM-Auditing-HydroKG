#!/bin/bash
#SBATCH -J HydroKG
#SBATCH -p main
#SBATCH --qos main
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G


#SBATCH -t 20:00:00
#SBATCH -o HydroKG_%j.out
#SBATCH -e HydroKG_%j.err

# --n_workers 8: matches -c 8 above. Parallelizes per-basin prediction generation and
# violation scanning (steps 3 and 4a) across that many workers within this one job --
# no SLURM array needed. If you change -c, change --n_workers to match.

set -eo pipefail   # stop on error; -u omitted since /etc/bashrc references unset vars

REPO_ROOT=/bighome/hdagne1/LSTM-Auditing-HydroKG
CONDA_ENV=BUDYKO-dev
RUN_DIR=external/HydroAuditToolFrameowrk/runs/run_0305_2015_seed658666
CAMELS_ROOT="$REPO_ROOT/data/CAMELS_US"

cd "$REPO_ROOT"
source ~/.bashrc
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

python scripts/run_enhanced_training.py \
  --run_dir "$RUN_DIR" \
  --camels_root "$CAMELS_ROOT" \
  --predictions_pickle "$RUN_DIR/lstm_seed658.p" \
  --n_epochs 3 \
  --n_workers 8 \
  --resume \
  --output_prefix results/hydrokg_run2

