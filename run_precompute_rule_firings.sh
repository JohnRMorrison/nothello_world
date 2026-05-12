#!/bin/bash
# Precompute cumulative rule-firings chunks alongside existing feature chunks.
# SLURM array of N tasks, each processes a range of pickle files.
#
# To process 120 pickle files in 20 chunks of 6 files each:
#   sbatch --array=0-19 run_precompute_rule_firings.sh
#
# Output: rule_firings_NNNN.npz where NNNN is the array task ID.

#SBATCH --job-name=rule_fire
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/rule_fire_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

# Each task processes FILES_PER_TASK pickle files
FILES_PER_TASK=6
START=$((SLURM_ARRAY_TASK_ID * FILES_PER_TASK))
END=$((START + FILES_PER_TASK))

echo "Task $SLURM_ARRAY_TASK_ID: pickle files [$START, $END)"

PYTHONUNBUFFERED=1 python precompute_rule_firings.py \
    --start-file $START --end-file $END \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --workers 16 --batch-size 2000
