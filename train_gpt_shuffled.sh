#!/bin/bash
# Train Othello-GPT with RANDOMIZED move order — tests whether the
# network can learn to predict next moves from order-free input.
# Usage:  sbatch train_gpt_shuffled.sh
# Override knobs:  EPOCHS=100 BATCH_SIZE=1024 sbatch train_gpt_shuffled.sh

#SBATCH --job-name=gpt_shuffle
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:8
#SBATCH --output=logs/train_gpt_shuffled_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Train Othello-GPT on shuffled-order games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "EPOCHS=${EPOCHS:-250}  BATCH_SIZE=${BATCH_SIZE:-4096}"
echo "Started at: $(date)"
echo "============================================"

# Default batch 4096 = 512 per GPU on 8 GPUs (matches the original Nanda recipe)
EPOCHS=${EPOCHS:-250} BATCH_SIZE=${BATCH_SIZE:-4096} NUM_WORKERS=${NUM_WORKERS:-16} \
    python train_gpt_shuffled.py

echo "Completed at: $(date)"
