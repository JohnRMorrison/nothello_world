#!/bin/bash
#SBATCH --job-name=delay_all
#SBATCH --output=logs/delay_all_%j.out
#SBATCH --error=logs/delay_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

# Generate games
python generate_variant_games.py \
    --variant delayed_flips \
    --num-games 2000000 \
    --output-dir experiments/variants/games_2m/delayed_flips \
    --seed 42

echo "Generation done at: $(date)"

# Fine-tune
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/delayed_flips \
    --output-dir experiments/variants/losses_2m \
    --label delayed_flips \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 8 \
    --batch-size 64

echo "FT done at: $(date)"

# Random init
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/delayed_flips \
    --output-dir experiments/variants/losses_2m_random \
    --label delayed_flips \
    --random-init \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
