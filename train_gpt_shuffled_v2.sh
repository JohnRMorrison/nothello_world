#!/bin/bash
# Corrected shuffled-input Othello-GPT training.
# Input: shuffled, parity-tagged prefix of a game.
# Target: the next move in the ORIGINAL game order.
# Usage:  sbatch train_gpt_shuffled_v2.sh
# Overrides: EPOCHS=100 BATCH_SIZE=4096 MAX_PREFIX_LEN=40

#SBATCH --job-name=gpt_shuf_v2
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:8
#SBATCH --output=logs/train_gpt_shuffled_v2_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Train GPT on shuffled parity-tagged prefix (predict next ORIGINAL move)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "EPOCHS=${EPOCHS:-50}  BATCH_SIZE=${BATCH_SIZE:-4096}  MAX_PREFIX_LEN=${MAX_PREFIX_LEN:-40}"
echo "Started at: $(date)"
echo "============================================"

EPOCHS=${EPOCHS:-50} BATCH_SIZE=${BATCH_SIZE:-4096} \
    NUM_WORKERS=${NUM_WORKERS:-16} MAX_PREFIX_LEN=${MAX_PREFIX_LEN:-40} \
    LOAD_CKPT=${LOAD_CKPT:-} \
    python train_gpt_shuffled_v2.py

echo "Completed at: $(date)"
