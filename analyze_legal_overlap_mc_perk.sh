#!/bin/bash
# Biased-MC legal-overlap analysis for a SINGLE k.  Submit one job per k
# to parallelize.
#
# Usage:
#   for k in 15 20 30 40 50; do
#       K=$k sbatch --job-name=overlap_mc_k$k analyze_legal_overlap_mc_perk.sh
#   done

#SBATCH --job-name=overlap_mc_k
#SBATCH -c 2
#SBATCH --time=03:00:00
#SBATCH --mem=8GB
#SBATCH --output=logs/overlap_mc_k_%j.out
#SBATCH --account=nklab

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

K=${K:?Must set K env var, e.g. K=20 sbatch analyze_legal_overlap_mc_perk.sh}

echo "============================================"
echo "Biased-MC legal-overlap: k=$K"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python analyze_legal_overlap_mc.py \
    --positions $K \
    --num-multisets ${NUM_MULTISETS:-50} \
    --samples-per-multiset ${SAMPLES_PER_MULTISET:-100}

echo "Completed at: $(date)"
