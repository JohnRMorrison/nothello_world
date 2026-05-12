#!/bin/bash
# Train a Nanda-style linear probe on the new H=1024 wheneven pattern
# detector's hidden layer. Mirrors the methodology that gave 99.25% on
# H=512 wheneven (probe_pattern_models.py: nn.Linear(H, 64*3), even/odd
# split, BCE loss, 5 epochs over all 12M games).
#
# Usage: sbatch run_probe_H1024_wheneven.sh

#SBATCH --job-name=probe_h1024_we
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_h1024_we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H1024_wheneven.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 1024 \
    --epochs 5
