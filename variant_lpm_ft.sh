#!/bin/bash
#SBATCH --job-name=lpm_ft
#SBATCH --output=logs/lpm_ft_%A_%a.out
#SBATCH --error=logs/lpm_ft_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-4

VARIANTS=(locked_flips no_diagonal_flips no_row_flips no_same_quadrant skip_empty_flips)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
GAMES_DIR="experiments/variants/games_2m/${VARIANT}"

echo "Variant: ${VARIANT} (FT, bs=16)"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/variants/losses_lpm \
    --label ${VARIANT} \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 1 \
    --batch-size 16

echo "Finished at: $(date)"
