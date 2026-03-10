#!/bin/bash
# Train MLP on 300-d features (180 base + 120 spatial) using streaming
# Usage: sbatch --array=0-1 spatial_features.sh
#
# Task 0: H=1024,  Task 1: H=2048

#SBATCH --job-name=spatial
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/spatial_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

H_ARRAY=(1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
H=${H_ARRAY[$TASK_ID]}

echo "============================================"
echo "Spatial Features (streaming): 300-d, H=$H, ~6M games"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train_streaming.py \
    --hidden $H \
    --epochs 10 \
    --spatial

echo "Completed at: $(date)"
