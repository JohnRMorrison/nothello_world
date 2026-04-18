#!/bin/bash
# Evaluate legal move prediction for all pattern detector models.
#
# 12 jobs: 8 trained models + 4 random projections
#
# Usage:
#   sbatch --array=0-11 eval_legal_moves.sh

#SBATCH --job-name=evlgl
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/evlgl_%A_%a.out
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
    0)  MODE="direct";    HIDDEN=512;  CKPT="pattern_simple_direct_H512.pt" ;;
    1)  MODE="direct";    HIDDEN=1024; CKPT="pattern_simple_direct_H1024.pt" ;;
    2)  MODE="emergent";  HIDDEN=512;  CKPT="pattern_simple_emergent_H512.pt" ;;
    3)  MODE="emergent";  HIDDEN=1024; CKPT="pattern_simple_emergent_H1024.pt" ;;
    4)  MODE="two-stage"; HIDDEN=512;  CKPT="pattern_simple_two-stage_H512.pt" ;;
    5)  MODE="two-stage"; HIDDEN=1024; CKPT="pattern_simple_two-stage_H1024.pt" ;;
    6)  MODE="e2e";       HIDDEN=512;  CKPT="pattern_simple_e2e_H512.pt" ;;
    7)  MODE="e2e";       HIDDEN=1024; CKPT="pattern_simple_e2e_H1024.pt" ;;
    8)  MODE="randproj";  HIDDEN=512;  CKPT="pattern_simple_randproj_s0_H512.pt" ;;
    9)  MODE="randproj";  HIDDEN=1024; CKPT="pattern_simple_randproj_s0_H1024.pt" ;;
    10) MODE="randproj";  HIDDEN=2048; CKPT="pattern_simple_randproj_s0_H2048.pt" ;;
    11) MODE="randproj";  HIDDEN=4096; CKPT="pattern_simple_randproj_s0_H4096.pt" ;;
esac

echo "============================================"
echo "Eval legal moves: mode=$MODE, H=$HIDDEN"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python eval_legal_moves.py \
    --ckpt "${CKPT_DIR}/${CKPT}" \
    --mode $MODE \
    --hidden $HIDDEN

echo "Completed at: $(date)"
