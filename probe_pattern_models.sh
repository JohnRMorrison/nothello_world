#!/bin/bash
# Probe hidden layers of all pattern detector models.
#
# 8 jobs matching train_pattern_simple.sh task layout:
#   0: direct    H=512     4: emergent H=1024
#   1: direct    H=1024    5: e2e      H=512
#   2: emergent  H=512     6: e2e      H=1024
#   3: two-stage H=512     7: two-stage H=1024
#
# Usage:
#   sbatch --array=0-7 probe_pattern_models.sh

#SBATCH --job-name=probe
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_%A_%a.out
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
    0) MODE="direct";    HIDDEN=512  ;;
    1) MODE="direct";    HIDDEN=1024 ;;
    2) MODE="emergent";  HIDDEN=512  ;;
    3) MODE="two-stage"; HIDDEN=512  ;;
    4) MODE="emergent";  HIDDEN=1024 ;;
    5) MODE="e2e";       HIDDEN=512  ;;
    6) MODE="e2e";       HIDDEN=1024 ;;
    7) MODE="two-stage"; HIDDEN=1024 ;;
esac

CKPT="${CKPT_DIR}/pattern_simple_${MODE}_H${HIDDEN}.pt"

echo "============================================"
echo "Probe: mode=$MODE, H=$HIDDEN"
echo "Checkpoint: $CKPT"
echo "Started at: $(date)"
echo "============================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT" \
    --mode $MODE \
    --hidden $HIDDEN

echo "Completed at: $(date)"
