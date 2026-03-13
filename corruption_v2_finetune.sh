#!/bin/bash
#SBATCH --job-name=rule_ft
#SBATCH --output=logs/rule_ft_%A_%a.out
#SBATCH --error=logs/rule_ft_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst

ALPHAS=(0 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
LABEL="alpha${ALPHA_STR}"
GAMES_DIR="experiments/corruption_v2/games/${LABEL}"

echo "============================================"
echo "Rule-based fine-tuning"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Alpha: ${ALPHA}, Label: ${LABEL}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/corruption_v2/losses \
    --label ${LABEL} \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 3 \
    --batch-size 64

echo "Finished at: $(date)"
