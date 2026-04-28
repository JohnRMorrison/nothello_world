#!/bin/bash
#SBATCH --job-name=incoh_rules
#SBATCH --output=logs/incoh_rules_%A_%a.out
#SBATCH --error=logs/incoh_rules_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source activate othello

echo "Condition: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"

python incoherent_rules_experiment.py \
    --condition-id $SLURM_ARRAY_TASK_ID \
    --output-dir experiments/incoherent_rules \
    --n-train 2000000 \
    --lr 5e-5 \
    --seed 42

echo "Finished at: $(date)"
