#!/bin/bash
#SBATCH --job-name=regen_adj
#SBATCH --output=logs/regen_adj_%j.out
#SBATCH --error=logs/regen_adj_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python regen_legal_moves.py \
    --variant adjacent_legal \
    --games-dir experiments/variants/games_2m/adjacent_legal \
    --workers 2

echo "Finished at: $(date)"
