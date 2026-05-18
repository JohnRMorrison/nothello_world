#!/bin/bash
# Train pattern detector H=512 with played+by_black (120-d).
# Minimal feature set: was-played indicator + which-color-placed.
# No timing info, no parity. Tests whether the model can do pattern
# detection from just the binary "where pieces are + their color".

#SBATCH --job-name=pat_h512_pb
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h512_pb_%j.out
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
    --features played+by_black
