#!/bin/bash
#SBATCH --job-name=ray_ctrl
#SBATCH --output=logs/ray_ctrl_%j.out
#SBATCH --error=logs/ray_ctrl_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst

echo "Started at: $(date)"

source activate othello

python experiment_ray_margin_control.py \
    --ray-margin-csv ray_margin_23k.csv \
    --output-csv ray_margin_control.csv \
    --output-summary ray_margin_control.txt

echo "Finished at: $(date)"
