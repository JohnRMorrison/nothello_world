#!/bin/bash
#SBATCH --job-name=cascade
#SBATCH --output=logs/cascade_%j.out
#SBATCH --error=logs/cascade_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

cd mechanistic_interpretability

START_LAYER=${START_LAYER:-4}

python multi_intervention.py \
    --probe-path main_linear_probe.pth \
    --n-games 200 \
    --cascade \
    --per-cell-scale \
    --scale 1.0 \
    --layer-intervene ${START_LAYER} \
    --probe-dir ../experiments/mathematical_transformation_experiments/heuristic_probe_results/probe_directions/probe_checkpoints \
    --n-values 1,2,3,8 \
    --output-dir ../experiments/multi_intervention_cascade_L${START_LAYER} \
    --save-probs \
    --seed 42

echo "Finished at: $(date)"
