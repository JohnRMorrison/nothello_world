#!/bin/bash
#SBATCH --job-name=multi_intv
#SBATCH --output=logs/multi_intv_%j.out
#SBATCH --error=logs/multi_intv_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

cd mechanistic_interpretability

python multi_intervention.py \
    --probe-path main_linear_probe.pth \
    --n-games 200 \
    --calibrate \
    --per-cell-scale \
    --probe-dir ../experiments/mathematical_transformation_experiments/heuristic_probe_results/probe_directions/probe_checkpoints \
    --n-values 1,2,3 \
    --output-dir ../experiments/multi_intervention \
    --seed 42

echo "Finished at: $(date)"
