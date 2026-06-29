#!/bin/bash
# One-time precompute of 960-d pattern legality for chunk_ext_*.npz.
#
# Usage:
#   sbatch --array=0-39%10 precompute_pattern_legality.sh
#   (Runs 40 tasks, max 10 concurrent — adjust %10 based on cluster load)

#SBATCH --job-name=precomp_patlegal
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/precomp_patlegal_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Precompute pattern legality, chunk $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 python precompute_pattern_legality.py

echo "Completed at: $(date)"
