#!/bin/bash
# Precompute legal move vectors for 6M games (12 chunks of 500K)
# Each array task handles one chunk independently
# Usage: sbatch --array=0-11 precompute_legal.sh

#SBATCH --job-name=legal
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --output=logs/precompute_legal_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================"
echo "Precomputing legal moves: chunk $TASK_ID (500K games)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE"
echo "Started at: $(date)"
echo "============================================"

python precompute_legal_moves.py \
    --max-games 6000000 \
    --n-workers 8 \
    --chunk-size 500000 \
    --chunk-id $TASK_ID

echo "Completed at: $(date)"
