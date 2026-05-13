#!/bin/bash
# Train pattern detector H=512 with when+by_black (120-d).
# Drops the `even` channel relative to the 180-d wheneven_byblack model,
# since by_black already encodes forfeit-correct color (more informative
# than even, which can't distinguish a player who passed a turn).
#
# Prereq: precompute_by_black.py must have completed (sidecar .npy files
# present alongside the chunk_NNNN.npz files).
#
# Usage: sbatch exp_h512_when_byblack.sh

#SBATCH --job-name=pat_h512_wb
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h512_wb_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_with_by_black.py \
    --hidden 512 \
    --epochs 3 \
    --features when+by_black
