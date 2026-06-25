#!/bin/bash
# Run analyze_order_via_db.py for a SINGLE k value.  Submit one job per k
# in parallel for 7-9x wall-clock speedup vs. doing them all in one job.
# Usage:
#   for k in 3 5 8 10 12 15 20 25 30; do
#       K=$k sbatch --job-name=order_db_k$k analyze_order_via_db_perk.sh
#   done

#SBATCH --job-name=order_db_k
#SBATCH -c 2
#SBATCH --time=04:00:00
#SBATCH --mem=60GB
#SBATCH --output=logs/order_db_k_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

K=${K:?Must set K env var, e.g. K=8 sbatch analyze_order_via_db_perk.sh}

echo "============================================"
echo "Order-ambiguity analysis: k=$K, full 20M games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python analyze_order_via_db.py \
    --num-games 20000000 \
    --num-pickle-files 240 \
    --positions $K

echo "Completed at: $(date)"
