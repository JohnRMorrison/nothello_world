#!/bin/bash
#SBATCH --job-name=sf_train
#SBATCH --output=logs/sf_train_%j.out
#SBATCH --error=logs/sf_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

# Fine-tune
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/self_flanking \
    --output-dir experiments/variants/losses_2m \
    --label self_flanking \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 8 \
    --batch-size 64

echo "FT done at: $(date)"

# Random init
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/self_flanking \
    --output-dir experiments/variants/losses_2m_random \
    --label self_flanking \
    --random-init \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
