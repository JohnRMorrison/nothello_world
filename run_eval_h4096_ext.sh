#!/bin/bash
# Evaluate the extended-range (turns 5-58) H=4096 movegrid probe:
#   1. Per-turn overall (all-64-cell) decoding accuracy
#   2. Per-cell 8x8 grid + per-region + worst cells
#
# Uses chunk_0039.npz (turns 5-53) + late_turns_eval.npz (turns 54-58)
# as eval data.
#
# Usage: sbatch run_eval_h4096_ext.sh

#SBATCH --job-name=eval_h4kx
#SBATCH -c 4
#SBATCH --time=01:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_h4kx_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs experiments/plots
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints
CHUNK_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks

PAT="$CKPT_DIR/pattern_simple_direct_H4096_move_grid_ext.pt"
PROBE="$CKPT_DIR/probe_direct_H4096_move_grid_ext_5_58.pt"
EVAL="$CHUNK_DIR/chunk_0039.npz"
EXTRA="$CHUNK_DIR/late_turns_eval.npz"

echo "=================================================================="
echo "Per-turn overall accuracy (turns 5-58)"
echo "=================================================================="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python plot_mlp_overall_by_turn.py \
    --pat-ckpt "$PAT" --probe-ckpt "$PROBE" \
    --hidden 4096 --features move_grid \
    --chunk-path "$EVAL" --extra-chunks "$EXTRA" \
    --output experiments/plots/mlp_h4096_ext_overall_by_turn.png

echo ""
echo "=================================================================="
echo "Per-cell 8x8 grid (turns 5-58)"
echo "=================================================================="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python analyze_mlp_probe_per_cell.py \
    --pat-ckpt "$PAT" --probe-ckpt "$PROBE" \
    --hidden 4096 --features move_grid \
    --chunk-path "$EVAL" --extra-chunks "$EXTRA" \
    --pos-start 5 --pos-end 59
