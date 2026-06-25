#!/bin/bash
# Analyze legal-move intersection/union across distinct boards reachable
# from the same (cell, parity) multiset.  One job per k.
#
# Usage:
#   for k in 5 8 10 12 15; do
#       K=$k sbatch --job-name=legal_overlap_k$k analyze_legal_overlap_perk.sh
#   done

#SBATCH --job-name=legal_overlap_k
#SBATCH -c 2
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/legal_overlap_k_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

K=${K:?Must set K env var, e.g. K=8 sbatch analyze_legal_overlap_perk.sh}

echo "============================================"
echo "Legal-overlap analysis: k=$K, full 20M games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python analyze_legal_overlap.py \
    --num-games 20000000 \
    --num-pickle-files 240 \
    --positions $K

echo "Completed at: $(date)"
