#!/bin/bash
#SBATCH --job-name=var_gen
#SBATCH --output=logs/var_gen_%A_%a.out
#SBATCH --error=logs/var_gen_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst

VARIANTS=(no_same_quadrant no_diagonal_flips no_row_flips locked_flips max_three_flips self_flanking delayed_flips)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Variant: ${VARIANT}"
echo "Started at: $(date)"

source activate othello

python generate_variant_games.py \
    --variant ${VARIANT} \
    --num-games 100000 \
    --output-dir experiments/variants/games_100k/${VARIANT} \
    --seed 42

echo "Finished at: $(date)"
