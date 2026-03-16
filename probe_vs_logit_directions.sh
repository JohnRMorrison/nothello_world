#!/bin/bash
#SBATCH --job-name=pvl_dirs
#SBATCH --output=logs/pvl_dirs_%j.out
#SBATCH --error=logs/pvl_dirs_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello 2>/dev/null || conda activate othello

cd mechanistic_interpretability

python probe_vs_logit_directions.py \
    --probe-dir ../experiments/mathematical_transformation_experiments/heuristic_probe_results/probe_directions/probe_checkpoints \
    --n-games 200 \
    --output-dir ../experiments/probe_vs_logit \
    --seed 42

echo "Finished at: $(date)"
