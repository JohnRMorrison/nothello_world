#!/bin/bash
#SBATCH --job-name=var_2m
#SBATCH --output=logs/var_2m_%A_%a.out
#SBATCH --error=logs/var_2m_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst

VARIANTS=(no_same_quadrant no_diagonal_flips no_row_flips locked_flips max_three_flips self_flanking delayed_flips)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Variant: ${VARIANT}"
echo "Started at: $(date)"

source activate othello

python generate_variant_games.py \
    --variant ${VARIANT} \
    --num-games 2000000 \
    --output-dir experiments/variants/games_2m/${VARIANT} \
    --seed 42

echo "Finished at: $(date)"
