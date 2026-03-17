#!/bin/bash
#SBATCH --job-name=traj_swap
#SBATCH --output=logs/traj_swap_%j.out
#SBATCH --error=logs/traj_swap_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python measure_trajectory_divergence.py \
    --variant swapped_colors \
    --std-games-dir experiments/corruption_v2/games_2m/alpha000 \
    --output-dir experiments/divergence/trajectory \
    --max-games 100000

echo "Finished at: $(date)"
