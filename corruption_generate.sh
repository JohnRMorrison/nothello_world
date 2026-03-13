#!/bin/bash
#SBATCH --job-name=cor_gen
#SBATCH --output=logs/corruption_gen_%A_%a.out
#SBATCH --error=logs/corruption_gen_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst

# 39 conditions: 3 types × 13 alphas
# Array index 0-38
# Index mapping: type = (idx // 13) + 1, alpha_idx = idx % 13

ALPHAS=(0 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

TYPE_IDX=$((SLURM_ARRAY_TASK_ID / 13))
ALPHA_IDX=$((SLURM_ARRAY_TASK_ID % 13))
TYPE=$((TYPE_IDX + 1))
ALPHA=${ALPHAS[$ALPHA_IDX]}

# Format alpha for directory name (e.g., 0.05 -> 005)
ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
LABEL="type${TYPE}_alpha${ALPHA_STR}"
OUTPUT_DIR="experiments/corruption/games/${LABEL}"

echo "============================================"
echo "Corruption game generation"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Type: ${TYPE}, Alpha: ${ALPHA}, Label: ${LABEL}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

# Skip alpha=0 duplicates (only generate once, for type1)
if [ "$ALPHA" = "0" ] && [ "$TYPE" != "1" ]; then
    echo "Skipping alpha=0 for type ${TYPE} (already generated for type 1)"
    # Create symlink instead
    mkdir -p experiments/corruption/games
    ln -sfn type1_alpha000 "experiments/corruption/games/${LABEL}"
    exit 0
fi

conda activate othello

python generate_corruption_games.py \
    --corruption-type ${TYPE} \
    --alpha ${ALPHA} \
    --num-games 100000 \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

echo "Finished at: $(date)"
