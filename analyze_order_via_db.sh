#!/bin/bash
# Run the database-driven order-ambiguity analysis on the full 20M-game
# synthetic dataset.  CPU-only (no GPU); the bottleneck is RAM (to hold
# all games + per-k dictionaries) and CPU time (replaying each game).
# Usage: sbatch analyze_order_via_db.sh

#SBATCH --job-name=order_db
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/order_db_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Order-ambiguity analysis: full 20M games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

# Positions to analyze.  Smaller k values have lots of repeated keys so the
# stats are clean; k=15-30 will have tens-to-hundreds of thousands of
# shared keys instead of 600.
python analyze_order_via_db.py \
    --num-games 20000000 \
    --num-pickle-files 240 \
    --positions 3 5 8 10 12 15 20 25 30

echo "Completed at: $(date)"
