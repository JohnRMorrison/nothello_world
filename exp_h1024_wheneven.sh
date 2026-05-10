#!/bin/bash
# Experiment 1: H=1024 + 120-d (when+even) for the 960-pattern model.
# Plain pattern BCE, train from scratch — capacity bump test.
#
# Usage: sbatch exp_h1024_wheneven.sh

#SBATCH --job-name=pat_h1024_we
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h1024_we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Experiment 1: H=1024 + when+even, 960-pattern BCE"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 1024 \
    --features when+even \
    --epochs 3

echo "Completed: $(date)"
