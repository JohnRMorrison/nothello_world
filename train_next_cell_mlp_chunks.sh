#!/bin/bash
# Train next-cell-prediction MLP on the precomputed fired_patterns_*.npz chunks.
#
# Usage:
#   sbatch train_next_cell_mlp_chunks.sh
#   HIDDEN=512 FEATURES=move_grid EPOCHS=3 MAX_CHUNKS=40 sbatch train_next_cell_mlp_chunks.sh

#SBATCH --job-name=next_cell_chunks
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/next_cell_chunks_%j.out
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
FEATURES=${FEATURES:-move_grid}
MAX_CHUNKS=${MAX_CHUNKS:-}
SEED=${SEED:-0}

echo "============================================"
echo "Train next-cell MLP on fired_patterns chunks"
echo "H=$HIDDEN  EPOCHS=$EPOCHS  FEATURES=$FEATURES  MAX_CHUNKS=$MAX_CHUNKS  SEED=$SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

MAX_CHUNKS_ARG=""
if [ -n "$MAX_CHUNKS" ]; then
    MAX_CHUNKS_ARG="--max-chunks $MAX_CHUNKS"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_next_cell_mlp_chunks.py \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --features $FEATURES \
    $MAX_CHUNKS_ARG \
    --seed $SEED

echo "Completed at: $(date)"
