#!/bin/bash
#SBATCH --job-name=rule_100k
#SBATCH --output=logs/rule_100k_%A_%a.out
#SBATCH --error=logs/rule_100k_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst

ALPHAS=(0 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
ALPHA_STR=$(printf "%03d" $(echo "$ALPHA * 100" | bc | cut -d. -f1))
OUTPUT_DIR="experiments/corruption_v2/games_100k/alpha${ALPHA_STR}"

echo "Alpha: ${ALPHA}, Label: alpha${ALPHA_STR}"
echo "Started at: $(date)"

source activate othello

python generate_rule_games.py \
    --alpha ${ALPHA} \
    --num-games 100000 \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

echo "Finished at: $(date)"
