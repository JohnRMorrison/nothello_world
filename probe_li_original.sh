#!/bin/bash
# ============================================================================
# Run Li et al.'s original train_probe_othello.py — one job per layer
# ============================================================================
#
# This is a diagnostic test: if Li's original code achieves ~95% accuracy
# with the same checkpoint, then there's a bug in our reimplementation.
# If it also gets ~76%, the issue is with the checkpoint or data.
#
# Usage:
#   sbatch probe_li_original.sh                # probe all layers 0-8
#   sbatch --array=6 probe_li_original.sh      # probe only layer 6
# ============================================================================

#SBATCH --job-name=probe_li
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/probe_li_%A_layer%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Load CUDA
module load cuda/11.8.0

# Activate conda environment
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs

cd $SLURM_SUBMIT_DIR

LAYER=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Probing layer: ${LAYER} (Li's original script)"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $LAYER --epo 16

echo "Completed at: $(date)"
