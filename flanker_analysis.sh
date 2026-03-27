#!/bin/bash
#SBATCH --job-name=flanker
#SBATCH --output=logs/flanker_%j.out
#SBATCH --error=logs/flanker_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source activate othello

echo "Started at: $(date)"

python flanker_analysis.py \
    --n-games 100000 \
    --max-per-rule 10000 \
    --output experiments/flanker_analysis.json \
    --plot experiments/flanker_analysis.png

echo "Finished at: $(date)"
