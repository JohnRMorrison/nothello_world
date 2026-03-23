#!/bin/bash
#SBATCH --job-name=nrules
#SBATCH --output=logs/nrules_%A_%a.out
#SBATCH --error=logs/nrules_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source activate othello

NRULES_LIST=(10 20 40 60 80)
NRULES=${NRULES_LIST[$SLURM_ARRAY_TASK_ID]}

echo "N_rules sweep: $NRULES rules, flip_color corruption"
echo "Started at: $(date)"

python sensitivity_param_search.py \
    --condition-id 3 \
    --output-dir experiments/nrules_sweep \
    --n-rules $NRULES \
    --lr 5e-5 \
    --seed 42

echo "Finished at: $(date)"
