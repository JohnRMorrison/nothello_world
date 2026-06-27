#!/bin/bash
# Train next-cell-prediction MLP with move_grid features.
# 60-way softmax over playable cells; CE loss against actual next move.
# Streams data from raw game pickles (no precomputed chunks needed).
#
# Usage:
#   sbatch train_next_cell_mlp.sh
#   MAX_GAMES=6000000 HIDDEN=512 sbatch train_next_cell_mlp.sh

#SBATCH --job-name=next_cell
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/next_cell_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

HIDDEN=${HIDDEN:-512}
EPOCHS=${EPOCHS:-3}
MAX_GAMES=${MAX_GAMES:-6000000}
SEED=${SEED:-0}

echo "============================================"
echo "Train next-cell MLP: H=$HIDDEN, EPOCHS=$EPOCHS, MAX_GAMES=$MAX_GAMES, SEED=$SEED"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_next_cell_mlp.py \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --max-games $MAX_GAMES \
    --seed $SEED

echo "Completed at: $(date)"
