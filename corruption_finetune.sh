#!/bin/bash
#SBATCH --job-name=cor_ft
#SBATCH --output=logs/corruption_ft_%A_%a.out
#SBATCH --error=logs/corruption_ft_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst

# Same 39 conditions as generation
ALPHAS=(0 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

TYPE_IDX=$((SLURM_ARRAY_TASK_ID / 13))
ALPHA_IDX=$((SLURM_ARRAY_TASK_ID % 13))
TYPE=$((TYPE_IDX + 1))
ALPHA=${ALPHAS[$ALPHA_IDX]}

ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
LABEL="type${TYPE}_alpha${ALPHA_STR}"
GAMES_DIR="experiments/corruption/games/${LABEL}"

echo "============================================"
echo "Corruption fine-tuning"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Type: ${TYPE}, Alpha: ${ALPHA}, Label: ${LABEL}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

conda activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/corruption/losses \
    --label ${LABEL} \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 3 \
    --batch-size 64

echo "Finished at: $(date)"
