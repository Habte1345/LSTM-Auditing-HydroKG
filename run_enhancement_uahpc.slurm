#!/bin/bash
#SBATCH -J HydroKG
#SBATCH -p main
#SBATCH --qos main
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -o HydroKG_%j.out
#SBATCH -e HydroKG_%j.err

# UAHPC SLURM submission template for the full enhancement pipeline.
# Submit with: sbatch scripts/run_enhancement_uahpc.slurm
# Edit REPO_ROOT, CONDA_ENV, RUN_DIR, and CAMELS_ROOT for your setup first.

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
  --output_prefix results/hydrokg_run1

# For GPU nodes instead, change the SBATCH directives above to:
#   #SBATCH -p gpu
#   #SBATCH --qos gpu
#   #SBATCH --gres=gpu:1
# and add --device cuda to the python command -- but confirm
# `python -c "import torch; print(torch.cuda.is_available())"` prints True first.
