#!/bin/bash
#SBATCH --job-name=new_sq
#SBATCH --output=logs/new_sq_%A_%a.out
#SBATCH --error=logs/new_sq_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source activate othello

echo "Condition: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"

python new_squares_experiment.py \
    --condition-id $SLURM_ARRAY_TASK_ID \
    --output-dir experiments/new_squares \
    --n-games 200000 \
    --lr 5e-5 \
    --bs 16 \
    --seed 42

echo "Finished at: $(date)"
