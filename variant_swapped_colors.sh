#!/bin/bash
#SBATCH --job-name=swapped
#SBATCH --output=logs/swapped_%j.out
#SBATCH --error=logs/swapped_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

# Generate games
python generate_variant_games.py \
    --variant swapped_colors \
    --num-games 2000000 \
    --output-dir experiments/variants/games_2m/swapped_colors \
    --seed 42

echo "Generation done at: $(date)"

# Fine-tune
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/swapped_colors \
    --output-dir experiments/variants/losses_2m \
    --label swapped_colors \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --epochs 8 \
    --batch-size 64

echo "FT done at: $(date)"

# Random init
python finetune_corruption.py \
    --games-dir experiments/variants/games_2m/swapped_colors \
    --output-dir experiments/variants/losses_2m_random \
    --label swapped_colors \
    --random-init \
    --epochs 8 \
    --batch-size 64

echo "Finished at: $(date)"
