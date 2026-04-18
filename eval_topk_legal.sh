#!/bin/bash
# Evaluate top-N legal accuracy for direct H=512 and H=1024.
#
# Usage:
#   sbatch --array=0-1 eval_topk_legal.sh

#SBATCH --job-name=topk
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/topk_%A_%a.out
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
esac

CKPT="${CKPT_DIR}/pattern_simple_direct_H${HIDDEN}.pt"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python eval_topk_legal.py \
    --ckpt "$CKPT" \
    --mode direct \
    --hidden $HIDDEN
