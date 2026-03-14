#!/bin/bash
#SBATCH --job-name=v2_rnd
#SBATCH --output=logs/v2_rnd_%A_%a.out
#SBATCH --error=logs/v2_rnd_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

VARIANTS=(adjacent_legal skip_empty_flips capture_any wrap_flips)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
GAMES_DIR="experiments/variants/games_2m/${VARIANT}"

echo "Variant: ${VARIANT} (random init)"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/variants/losses_2m_random \
    --label ${VARIANT} \
    --random-init \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
