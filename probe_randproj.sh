#!/bin/bash
# Probe hidden layers of random projection pattern detector models.
#
# 4 jobs: H=512, 1024, 2048, 4096
#
# Usage:
#   sbatch --array=0-3 probe_randproj.sh

#SBATCH --job-name=prrp
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/prrp_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=${SLURM_ARRAY_TASK_ID:-0}
CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

case $TASK in
    0) HIDDEN=512  ;;
    1) HIDDEN=1024 ;;
    2) HIDDEN=2048 ;;
    3) HIDDEN=4096 ;;
esac

CKPT="${CKPT_DIR}/pattern_simple_randproj_s0_H${HIDDEN}.pt"

echo "============================================"
echo "Probe randproj: H=$HIDDEN"
echo "Checkpoint: $CKPT"
echo "Started at: $(date)"
echo "============================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT" \
    --mode direct \
    --hidden $HIDDEN \
    --epochs 2

echo "Completed at: $(date)"
