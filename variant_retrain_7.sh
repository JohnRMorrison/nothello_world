#!/bin/bash
#SBATCH --job-name=retrain7
#SBATCH --output=logs/retrain7_%A_%a.out
#SBATCH --error=logs/retrain7_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

VARIANTS=(no_diagonal_flips no_row_flips locked_flips max_three_flips self_flanking delayed_flips no_same_quadrant)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
GAMES_DIR="experiments/variants/games_2m/${VARIANT}"

echo "Variant: ${VARIANT}"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/variants/losses_2m \
    --label ${VARIANT} \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
