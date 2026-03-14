#!/bin/bash
#SBATCH --job-name=mlp_intv
#SBATCH --output=logs/mlp_intv_%j.out
#SBATCH --error=logs/mlp_intv_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

cd mechanistic_interpretability

python multi_intervention.py \
    --mlp-baseline \
    --mlp-ckpt ../experiments/mathematical_transformation_experiments/heuristic_probe_results/mlp_checkpoints/mlp_180_H1024.pt \
    --n-games 10 \
    --calibrate \
    --output-dir ../experiments/multi_intervention_mlp \
    --seed 42

echo "Finished at: $(date)"
