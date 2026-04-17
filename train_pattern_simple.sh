#!/bin/bash
# Train pattern detectors — simple version.
#
# 6 jobs (sbatch --array=0-5):
#   0: direct    H=512     3: emergent H=512
#   1: direct    H=1024    4: emergent H=1024
#   2: e2e       H=512     5: e2e      H=1024
#
# Usage:
#   sbatch --array=0-5 --time=12:00:00 train_pattern_simple.sh

#SBATCH --job-name=patsim
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/patsim_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=${SLURM_ARRAY_TASK_ID:-0}

case $TASK in
    0) MODE="direct";   HIDDEN=512  ;;
    1) MODE="direct";   HIDDEN=1024 ;;
    2) MODE="e2e";      HIDDEN=512  ;;
    3) MODE="emergent"; HIDDEN=512  ;;
    4) MODE="emergent"; HIDDEN=1024 ;;
    5) MODE="e2e";      HIDDEN=1024 ;;
esac

echo "============================================"
echo "Pattern simple: mode=$MODE, H=$HIDDEN"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode $MODE \
    --hidden $HIDDEN \
    --epochs 3

echo "Completed at: $(date)"
