#!/bin/bash
#SBATCH --job-name=adj_div
#SBATCH --output=logs/adj_div_%j.out
#SBATCH --error=logs/adj_div_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python measure_legal_divergence.py \
    --games-dir experiments/variants/games_2m/adjacent_legal \
    --condition-type variant \
    --condition-label adjacent_legal \
    --variant-name adjacent_legal \
    --std-games-dir experiments/corruption_v2/games_2m/alpha000 \
    --output-dir experiments/divergence \
    --max-games 100000

echo "Finished at: $(date)"
