#!/bin/bash
#SBATCH --job-name=sens_gen
#SBATCH --output=logs/sens_gen_%A_%a.out
#SBATCH --error=logs/sens_gen_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-8

# 3x3 factorial: corruption_type x sensitivity_rank
CORRUPTION_TYPES=(full full full terminal_only terminal_only terminal_only drop_opponent drop_opponent drop_opponent)
SENSITIVITY_RANKS=(high low random high low random high low random)
DIR_NAMES=(full_high full_low full_random terminal_high terminal_low terminal_random drop_high drop_low drop_random)

CT=${CORRUPTION_TYPES[$SLURM_ARRAY_TASK_ID]}
SR=${SENSITIVITY_RANKS[$SLURM_ARRAY_TASK_ID]}
DN=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Condition: ${CT} x ${SR} (${DN})"
echo "Started at: $(date)"

source activate othello

python generate_sensitivity_games.py \
    --corruption-type ${CT} \
    --sensitivity-rank ${SR} \
    --target-divergence 0.20 \
    --sensitivity-file behavioral_data/sensitivity.json \
    --num-games 2000000 \
    --output-dir behavioral_data/games/${DN}/ \
    --seed 42

echo "Finished at: $(date)"
