#!/bin/bash
# Train pattern detector H=4096 with move_grid (3600-d) input.
# Tests whether MASSIVE first-layer capacity (3600 x 4096 = 14.7M params)
# combined with rich input is enough to overcome the 1-layer flip-tracking
# limit. Compare to H=512 move_grid (existing) and H=2048 wheneven (in flight).
#
# Usage: sbatch exp_h4096_movegrid.sh

#SBATCH --job-name=pat_h4k_mg
#SBATCH -c 4
#SBATCH --time=18:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h4k_mg_%j.out
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
    --hidden 4096 \
    --features move_grid \
    --epochs 3
