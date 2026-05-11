#!/bin/bash
# Step 0 for the class-imbalance hypothesis: measure per-pattern firing rates
# bucketed by target-cell class. Reads chunk_0039.npz, deletes the feature
# array immediately so we can run on a modest allocation.
#
# Usage: sbatch run_pattern_base_rates.sh

#SBATCH --job-name=pat_rates
#SBATCH -c 2
#SBATCH --time=00:15:00
#SBATCH --mem=20GB
#SBATCH --output=logs/pat_rates_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Per-pattern firing rates by target cell class"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python pattern_base_rates.py

echo "Completed: $(date)"
