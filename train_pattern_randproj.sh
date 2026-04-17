#!/bin/bash
# Random projection pattern detectors: frozen random first layer + trained output.
#
# 4 jobs: H=512, 1024, 2048, 4096
#
# Usage:
#   sbatch --array=0-3 train_pattern_randproj.sh

#SBATCH --job-name=patrp
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/patrp_%A_%a.out
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
    0) HIDDEN=512  ;;
    1) HIDDEN=1024 ;;
    2) HIDDEN=2048 ;;
    3) HIDDEN=4096 ;;
esac

echo "============================================"
echo "Pattern randproj: H=$HIDDEN"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode randproj \
    --hidden $HIDDEN \
    --epochs 3 \
    --seed 0

echo "Completed at: $(date)"
