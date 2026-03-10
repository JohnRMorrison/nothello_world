#!/bin/bash
# Scaling analysis: accuracy vs number of training games
# Usage: sbatch --array=0-5 game_scaling.sh
#
# Task 0: 5K,  1: 10K,  2: 50K,  3: 100K,  4: 250K,  5: 500K

#SBATCH --job-name=scaling
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/scaling_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

GAMES_ARRAY=(5000 10000 50000 100000 250000 500000)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_GAMES=${GAMES_ARRAY[$TASK_ID]}

echo "============================================"
echo "Game Scaling: $N_GAMES games, H=1024"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python feature_ablation.py \
    --subset-id 6 \
    --max-games $N_GAMES \
    --epochs 10 \
    --hidden 1024 \
    --precomputed

echo "Completed at: $(date)"
