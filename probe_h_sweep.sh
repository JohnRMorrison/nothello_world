#!/bin/bash
# Train board-decoding probes on existing pattern detectors at H=1024 and H=4096
# with uniform-turn eval. Compared to H=512 wheneven (91.25%) and H=512
# move_grid (91.99%) to measure the width-vs-decoding-accuracy ceiling.
#
# Array tasks:
#   0 = H=1024 wheneven (60-d when input)
#   1 = H=4096 move_grid (3600-d input — slower, may need full 24h)
#
# Usage: sbatch --array=0-1 probe_h_sweep.sh

#SBATCH --job-name=probe_hsw
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_hsw_%A_%a.out
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
        echo "Probe: H=1024 wheneven (60-d when)"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
            --ckpt $CKPT_DIR/pattern_simple_direct_H1024_wheneven.pt \
            --mode direct --hidden 1024 --epochs 5
        ;;
    1)
        echo "Probe: H=4096 move_grid (3600-d input)"
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
            --ckpt $CKPT_DIR/pattern_simple_direct_H4096_move_grid.pt \
            --mode direct --hidden 4096 --epochs 5
        ;;
esac
