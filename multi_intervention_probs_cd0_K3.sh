#!/bin/bash
#SBATCH --job-name=intv_cd0K3
#SBATCH --output=logs/intv_cd0K3_%j.out
#SBATCH --error=logs/intv_cd0K3_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# OGPT intervention experiment: N=1, cd0 local calibration, K=3 magnitude.
# Multiplies cd0's per-cell adaptive magnitudes by 3.

echo "Started at: $(date)"

source activate othello

cd mechanistic_interpretability

LAYER=${LAYER:-5}

python multi_intervention.py \
    --probe-path main_linear_probe.pth \
    --n-games 200 \
    --calibrate \
    --per-cell-scale \
    --layer-intervene ${LAYER} \
    --cal-depth 0 \
    --scale 3.0 \
    --probe-dir ../experiments/mathematical_transformation_experiments/heuristic_probe_results/probe_directions/probe_checkpoints \
    --n-values 1 \
    --output-dir ../experiments/multi_intervention_probs_cd0_K3 \
    --save-probs \
    --seed 42

echo "Finished at: $(date)"
