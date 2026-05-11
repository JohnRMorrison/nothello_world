#!/bin/bash
# CCGP on OGPT residual stream. Default layer 4; can sweep multiple layers
# by submitting multiple jobs with --layer N.
#
# Usage:
#   sbatch run_ccgp_ogpt.sh           # defaults to layer 4
#   sbatch run_ccgp_ogpt.sh 2         # override layer
#   sbatch run_ccgp_ogpt.sh 6
#   for L in 2 4 6; do sbatch run_ccgp_ogpt.sh $L; done

#SBATCH --job-name=ccgp_ogpt
#SBATCH -c 4
#SBATCH --time=01:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ccgp_ogpt_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

LAYER=${1:-4}

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python compute_ccgp_ogpt.py \
    --layer "$LAYER" \
    --n-games 3000 \
    --ccgp-mode both
