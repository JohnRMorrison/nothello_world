#!/bin/bash
#SBATCH --job-name=sens_ft
#SBATCH --output=logs/sens_ft_%A_%a.out
#SBATCH --error=logs/sens_ft_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-17

# 9 conditions x 2 (FT/RND) = 18 runs
# Even indices = FT, odd indices = RND
DIR_NAMES=(full_high full_high full_low full_low full_random full_random terminal_high terminal_high terminal_low terminal_low terminal_random terminal_random drop_high drop_high drop_low drop_low drop_random drop_random)

IDX=$SLURM_ARRAY_TASK_ID
DN=${DIR_NAMES[$IDX]}
IS_RND=$((IDX % 2))

if [ $IS_RND -eq 0 ]; then
    MODE="FT"
    EXTRA_ARGS="--ckpt ckpts/gpt_synthetic.ckpt"
    OUTPUT_DIR="behavioral_data/losses_sensitivity"
else
    MODE="RND"
    EXTRA_ARGS="--random-init"
    OUTPUT_DIR="behavioral_data/losses_sensitivity_random"
fi

echo "Condition: ${DN} (${MODE})"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir behavioral_data/games/${DN}/ \
    --output-dir ${OUTPUT_DIR} \
    --label ${DN} \
    ${EXTRA_ARGS} \
    --epochs 1 \
    --batch-size 16

echo "Finished at: $(date)"
