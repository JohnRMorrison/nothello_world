#!/bin/bash
# Test the compressibility hypothesis for a single k.
# Submit one job per k for parallelism.
#
# Usage:
#   for k in 15 20 30 40 50; do
#       K=$k sbatch --job-name=compress_k$k analyze_overlap_compressibility_perk.sh
#   done

#SBATCH --job-name=compress_k
#SBATCH -c 2
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
#SBATCH --output=logs/compress_k_%j.out
#SBATCH --account=nklab

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

K=${K:?Must set K env var, e.g. K=20 sbatch analyze_overlap_compressibility_perk.sh}

echo "============================================"
echo "Compressibility analysis: k=$K"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python analyze_overlap_compressibility.py \
    --positions $K \
    --num-multisets ${NUM_MULTISETS:-500} \
    --samples-per-multiset ${SAMPLES_PER_MULTISET:-30}

echo "Completed at: $(date)"
