#!/bin/bash
# Precompute per-turn fired patterns chunks for next-rule prediction.
#
# To process 120 pickle files in 20 chunks of 6 files each (throttled):
#   sbatch --array=0-39%20 run_precompute_fired_patterns.sh
#
# Output: fired_patterns_NNNN.npz per chunk-id.

#SBATCH --job-name=fired_pat
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/fired_pat_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

FILES_PER_TASK=6
START=$((SLURM_ARRAY_TASK_ID * FILES_PER_TASK))
END=$((START + FILES_PER_TASK))

echo "Task $SLURM_ARRAY_TASK_ID: pickle files [$START, $END)"

PYTHONUNBUFFERED=1 python precompute_fired_patterns.py \
    --start-file $START --end-file $END \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --workers 16 --batch-size 2000
