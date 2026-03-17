#!/bin/bash
#SBATCH --job-name=game_stats
#SBATCH --output=logs/game_stats_%j.out
#SBATCH --error=logs/game_stats_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python compute_game_statistics.py \
    --batch \
    --variant-base experiments/variants/games_2m \
    --corruption-base experiments/corruption_v2/games_2m \
    --output-dir experiments/divergence/game_stats \
    --max-games 100000

echo "Finished at: $(date)"
