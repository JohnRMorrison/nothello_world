#!/bin/bash
# H=2048 + when+even, single-layer direct mode pattern detector.
# Tests whether the recall@K ceiling is a width issue at the current
# architecture (1 hidden layer, stays inside the "no world model" control).
#
# Usage: sbatch exp_h2048_wheneven.sh

#SBATCH --job-name=pat_h2048
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h2048_we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 2048 \
    --features when+even \
    --epochs 3
