#!/bin/bash
#SBATCH --job-name=imp_ft
#SBATCH --output=logs/imp_ft_%A_%a.out
#SBATCH --error=logs/imp_ft_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-5

GERS=(0.0 0.1 0.2 0.3 0.4 0.5)

GER=${GERS[$SLURM_ARRAY_TASK_ID]}
GER_LABEL=$(printf "ger%03d" $(echo "$GER * 100" | bc | cut -d. -f1))
GAMES_DIR="experiments/impossible/games_2m/${GER_LABEL}"

echo "GER: ${GER} (FT, bs=16)"
echo "Started at: $(date)"

source activate othello

python finetune_corruption.py \
    --games-dir ${GAMES_DIR} \
    --output-dir experiments/impossible/losses_lpm \
    --label ${GER_LABEL} \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 1 \
    --batch-size 16

echo "Finished at: $(date)"
