#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --output=logs/early_dyn_%A_%a.out
#SBATCH --error=logs/early_dyn_%A_%a.err
#SBATCH --job-name=early_dyn

source activate othello

CONDITIONS=(
    fm_full_high fm_full_low
    fm_terminal_high fm_terminal_low
    fm_drop_high fm_drop_low
)

COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
echo "Condition: ${COND} (FT)"

python finetune_corruption.py \
    --games-dir behavioral_data/games/${COND} \
    --output-dir behavioral_data/early_dynamics \
    --label "${COND}_ft" \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 1 \
    --batch-size 16 \
    --eval-every 5 \
    --max-steps 200
