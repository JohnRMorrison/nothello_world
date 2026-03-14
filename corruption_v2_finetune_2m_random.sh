#!/bin/bash
#SBATCH --job-name=rule_rnd
#SBATCH --output=logs/rule_rnd_%A_%a.out
#SBATCH --error=logs/rule_rnd_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

ALPHAS=(0 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
LABEL="alpha${ALPHA_STR}"
GAMES_DIR="experiments/corruption_v2/games_2m/${LABEL}"

echo "Alpha: ${ALPHA}, Label: ${LABEL} (random init)"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/corruption_v2/losses_2m_random \
    --label ${LABEL} \
    --random-init \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
