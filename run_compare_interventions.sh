#!/bin/bash
# Side-by-side intervention comparison: MLP vs OGPT on identical positions.
# 100k games x 10 positions x 3 directions x up to 2 targets each
# ~= up to 6M paired interventions. Long-running job.
#
# Usage: sbatch run_compare_interventions.sh

#SBATCH --job-name=cmp_intv
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cmp_intv_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python compare_interventions.py \
    --n-games 100000 \
    --positions-per-game 10 \
    --scale 3 \
    --output logs/compare_interventions.npz
