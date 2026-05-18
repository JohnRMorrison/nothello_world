#!/bin/bash
# Train next-rule prediction MLP on the per-turn fired-pattern chunks.
#
# Prereq: run_precompute_fired_patterns.sh must have produced
#         fired_patterns_NNNN.npz chunks first.
#
# Usage:
#   sbatch run_train_next_rule.sh
#   # Or with a dependency:
#   sbatch --dependency=afterok:<PRECOMPUTE_JOB_ID> run_train_next_rule.sh

#SBATCH --job-name=next_rule
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/next_rule_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_next_rule.py \
    --hidden 512 --epochs 3 --features when+even
