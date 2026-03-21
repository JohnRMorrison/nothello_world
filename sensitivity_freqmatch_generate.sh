#!/bin/bash
#SBATCH --job-name=sfm_gen
#SBATCH --output=logs/sfm_gen_%A_%a.out
#SBATCH --error=logs/sfm_gen_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-5

# 3 corruption types x 2 sensitivity ranks (high/low only, no random)
# Random doesn't apply to frequency matching
CORRUPTION_TYPES=(full full terminal_only terminal_only drop_opponent drop_opponent)
SENSITIVITY_RANKS=(high low high low high low)
DIR_NAMES=(fm_full_high fm_full_low fm_terminal_high fm_terminal_low fm_drop_high fm_drop_low)

CT=${CORRUPTION_TYPES[$SLURM_ARRAY_TASK_ID]}
SR=${SENSITIVITY_RANKS[$SLURM_ARRAY_TASK_ID]}
DN=${DIR_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Condition: ${CT} x ${SR} (${DN}), frequency-matched"
echo "Started at: $(date)"

source activate othello

python generate_sensitivity_games.py \
    --corruption-type ${CT} \
    --sensitivity-rank ${SR} \
    --fixed-count 100 \
    --frequency-matched \
    --sensitivity-file behavioral_data/sensitivity.json \
    --num-games 2000000 \
    --output-dir behavioral_data/games/${DN}/ \
    --seed 42

echo "Finished at: $(date)"
