#!/bin/bash
# Train Nanda-style board-state probe on by_black-input pattern detectors.
# Array tasks:
#   0 = wheneven_byblack (180-d, existing checkpoint)
#   1 = when_byblack     (120-d, depends on pat_h512_wb training)
#   2 = move_grid        (3600-d, existing checkpoint; uses probe_pattern_models.py)
#
# Usage: sbatch --array=0-2 probe_h512_byblack.sh

#SBATCH --job-name=probe_byb
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_byb_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Probe: when+even+by_black (180-d, existing wheneven_byblack ckpt)"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_with_by_black.py \
            --ckpt $CKPT_DIR/pattern_simple_direct_H512_wheneven_byblack.pt \
            --features when+even+by_black \
            --hidden 512 --epochs 5
        ;;
    1)
        echo "Probe: when+by_black (120-d, when_byblack ckpt)"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_with_by_black.py \
            --ckpt $CKPT_DIR/pattern_simple_direct_H512_when_byblack.pt \
            --features when+by_black \
            --hidden 512 --epochs 5
        ;;
    2)
        echo "Probe: move_grid (3600-d, existing move_grid ckpt) via probe_pattern_models"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
            --ckpt $CKPT_DIR/pattern_simple_direct_H512_move_grid.pt \
            --mode direct --hidden 512 --epochs 5
        ;;
esac
