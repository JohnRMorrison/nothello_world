#!/bin/bash
# Precompute 60-d 'when' feature chunks (parallel).
#
# Usage: sbatch --array=0-39 precompute_when60.sh
#
# Each task processes one chunk. No GPU needed.

#SBATCH --job-name=when60
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --mem=30GB
#SBATCH --output=logs/when60_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================"
echo "Precompute when60: chunk $CHUNK_ID"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $CHUNK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python precompute_when60.py --chunk-id $CHUNK_ID

echo "Completed at: $(date)"
