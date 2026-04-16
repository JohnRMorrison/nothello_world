#!/bin/bash
# Train 960 pattern detectors on MLP-predicted board state.
#
# 8 jobs (sbatch --array=0-7):
#   Task 0: two-stage,   H=512
#   Task 1: two-stage,   H=1024
#   Task 2: end-to-end,  H=512
#   Task 3: end-to-end,  H=1024
#   Task 4: direct,      H=512
#   Task 5: direct,      H=1024
#   Task 6: emergent,    H=512    (E2E architecture, no board aux loss)
#   Task 7: emergent,    H=1024
#
# Prerequisites: feature chunks 0-39 precomputed (12M games).
#   If only chunks 0-19 exist, run: sbatch --array=20-39 precompute_12m.sh
#
# Usage:
#   sbatch --array=0-7 train_pattern_det.sh

#SBATCH --job-name=patdet
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/patdet_%A_%a.out
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
    0) MODE="two-stage";  HIDDEN=512  ;;
    1) MODE="two-stage";  HIDDEN=1024 ;;
    2) MODE="end-to-end"; HIDDEN=512  ;;
    3) MODE="end-to-end"; HIDDEN=1024 ;;
    4) MODE="direct";     HIDDEN=512  ;;
    5) MODE="direct";     HIDDEN=1024 ;;
    6) MODE="emergent";   HIDDEN=512  ;;
    7) MODE="emergent";   HIDDEN=1024 ;;
esac

echo "============================================"
echo "Pattern detectors: mode=$MODE, H=$HIDDEN"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train_pattern_detectors.py \
    --mode $MODE \
    --hidden $HIDDEN \
    --epochs 3

echo "Completed at: $(date)"
