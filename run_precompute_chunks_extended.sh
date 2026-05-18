#!/bin/bash
# Precompute extended-range chunks (turns 5-58) for fair late-game eval.
# Mirrors the existing chunk_NNNN.npz format. Each task processes 6
# pickle files; output is chunk_ext_NNNN.npz.
#
# Usage:
#   sbatch --array=0-39%20 run_precompute_chunks_extended.sh

#SBATCH --job-name=chunk_ext
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/chunk_ext_%A_%a.out
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

PYTHONUNBUFFERED=1 python precompute_chunks_extended.py \
    --start-file $START --end-file $END \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --pos-start 5 --pos-end 59
