#!/bin/bash
#SBATCH --job-name=sfm_ft
#SBATCH --output=logs/sfm_ft_%A_%a.out
#SBATCH --error=logs/sfm_ft_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-11

# 6 conditions x 2 (FT/RND) = 12 runs
DIR_NAMES=(fm_full_high fm_full_high fm_full_low fm_full_low fm_terminal_high fm_terminal_high fm_terminal_low fm_terminal_low fm_drop_high fm_drop_high fm_drop_low fm_drop_low)

IDX=$SLURM_ARRAY_TASK_ID
DN=${DIR_NAMES[$IDX]}
IS_RND=$((IDX % 2))

if [ $IS_RND -eq 0 ]; then
    MODE="FT"
    EXTRA_ARGS="--ckpt ckpts/gpt_synthetic.ckpt"
    OUTPUT_DIR="behavioral_data/losses_freqmatch"
else
    MODE="RND"
    EXTRA_ARGS="--random-init"
    OUTPUT_DIR="behavioral_data/losses_freqmatch_random"
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
