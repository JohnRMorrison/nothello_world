#!/bin/bash
# Precompute legal move vectors for 6M games
# CPU-only, needs many cores and memory for the output arrays
# Usage: sbatch precompute_legal.sh

#SBATCH --job-name=legal
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/precompute_legal_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Precomputing legal move vectors (6M games)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE"
echo "Started at: $(date)"
echo "============================================"

python precompute_legal_moves.py \
    --max-games 6000000 \
    --n-workers 16 \
    --chunk-size 500000

echo "Completed at: $(date)"
