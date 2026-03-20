#!/bin/bash
#SBATCH --job-name=beh_fit
#SBATCH --output=logs/beh_fit_%A_%a.out
#SBATCH --error=logs/beh_fit_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-59

echo "Cell: ${SLURM_ARRAY_TASK_ID}"
echo "Started at: $(date)"

source activate othello

python behavioral_fit.py \
    --cell ${SLURM_ARRAY_TASK_ID} \
    --data-dir behavioral_data \
    --output-dir behavioral_data

echo "Finished at: $(date)"
