#!/bin/bash
# Sweep manual scale multipliers along the cd0 (L5-native) probe direction.
# K=1.0 reproduces the existing cd0 run; K>1 amplifies the same direction.
#
# Usage: sbatch --array=0-6 cd0_scale_sweep.sh
#SBATCH --job-name=cd0_sweep
#SBATCH --output=logs/cd0_sweep_%A_%a.out
#SBATCH --error=logs/cd0_sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst

source activate othello

KS=(0.5 1.0 2.0 3.0 5.0 8.0 12.0)
K=${KS[$SLURM_ARRAY_TASK_ID]}

echo "Scale multiplier K = $K"
echo "Started at $(date)"

cd /engram/nklab/jrm2182/nothello_world

python mechanistic_interpretability/multi_intervention.py \
    --probe-dir mechanistic_interpretability/probe_checkpoints \
    --layer-intervene 5 \
    --scale "$K" \
    --per-cell-scale \
    --cal-depth 0 \
    --n-values 1 \
    --n-games 200 \
    --save-probs \
    --output-dir experiments/cd0_scale_sweep/k_${K}

echo "Finished at $(date)"
