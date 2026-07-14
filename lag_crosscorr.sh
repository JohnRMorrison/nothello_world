#!/bin/bash
#SBATCH --job-name=lag_xcorr
#SBATCH --output=logs/lag_xcorr_%j.out
#SBATCH --error=logs/lag_xcorr_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python experiment_lag_crosscorr.py \
    --output-csv lag_crosscorr.csv \
    --output-summary lag_crosscorr.txt

echo "Finished at: $(date)"
