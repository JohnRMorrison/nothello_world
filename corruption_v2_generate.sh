#!/bin/bash
#SBATCH --job-name=rule_gen
#SBATCH --output=logs/rule_gen_%A_%a.out
#SBATCH --error=logs/rule_gen_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst

# 10 alpha values, array index 0-9
ALPHAS=(0 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
OUTPUT_DIR="experiments/corruption_v2/games/alpha${ALPHA_STR}"

echo "============================================"
echo "Rule-based game generation"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Alpha: ${ALPHA}, Label: alpha${ALPHA_STR}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

source activate othello

python generate_rule_games.py \
    --alpha ${ALPHA} \
    --num-games 2000000 \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

echo "Finished at: $(date)"
