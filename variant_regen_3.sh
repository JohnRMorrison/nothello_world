#!/bin/bash
#SBATCH --job-name=regen3
#SBATCH --output=logs/regen3_%A_%a.out
#SBATCH --error=logs/regen3_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

VARIANTS=(self_flanking delayed_flips max_three_flips)

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
