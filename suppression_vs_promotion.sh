#!/bin/bash
#SBATCH --job-name=supp_prom
#SBATCH --output=logs/supp_prom_%A_%a.out
#SBATCH --error=logs/supp_prom_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source activate othello

echo "Condition: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"

python suppression_vs_promotion.py \
    --condition-id $SLURM_ARRAY_TASK_ID \
    --output-dir experiments/supp_vs_prom \
    --n-train 200000 \
    --lr 5e-5 \
    --seed 42

echo "Finished at: $(date)"
