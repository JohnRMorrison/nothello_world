#!/bin/bash
# Stream-train MLP on all available feature chunks.
# Run after precompute_more.sh completes.
#
# Usage: sbatch train_streaming.sh [features] [hidden] [epochs]
#   e.g.: sbatch train_streaming.sh all 1024 10
#         sbatch train_streaming.sh when+even 1024 10

#SBATCH --job-name=mlp_strm
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_stream_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

FEATURES=${1:-all}
HIDDEN=${2:-1024}
EPOCHS=${3:-10}

echo "============================================"
echo "Streaming MLP: features=$FEATURES H=$HIDDEN epochs=$EPOCHS"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train_streaming.py \
    --features $FEATURES \
    --hidden $HIDDEN \
    --epochs $EPOCHS

echo "Completed at: $(date)"
