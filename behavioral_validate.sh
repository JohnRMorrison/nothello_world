#!/bin/bash
#SBATCH --job-name=beh_val
#SBATCH --output=logs/beh_validate_%j.out
#SBATCH --error=logs/beh_validate_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python behavioral_validate.py \
    --data-dir behavioral_data \
    --policy-games 10000 \
    --dist-positions 10000 \
    --coverage-positions 100000

echo "Finished at: $(date)"
