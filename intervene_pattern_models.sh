#!/bin/bash
# Run Nanda-style interventions on pattern detector MLPs.
#
# 2 jobs: direct H=512, direct H=1024
# (Can extend to other modes later)
#
# Usage:
#   sbatch --array=0-1 intervene_pattern_models.sh

#SBATCH --job-name=intv
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/intv_%A_%a.out
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
    0) MODE="direct"; HIDDEN=512  ;;
    1) MODE="direct"; HIDDEN=1024 ;;
esac

MODEL_CKPT="${CKPT_DIR}/pattern_simple_${MODE}_H${HIDDEN}.pt"
PROBE_CKPT="${CKPT_DIR}/probe_${MODE}_H${HIDDEN}.pt"

echo "============================================"
echo "Intervention: mode=$MODE, H=$HIDDEN"
echo "Model: $MODEL_CKPT"
echo "Probe: $PROBE_CKPT"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python intervene_pattern_models.py \
    --model-ckpt "$MODEL_CKPT" \
    --probe-ckpt "$PROBE_CKPT" \
    --mode $MODE \
    --hidden $HIDDEN \
    --n-games 500

echo "Completed at: $(date)"
