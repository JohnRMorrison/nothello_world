#!/bin/bash
# v3: shuffled context + 60 simultaneous query positions via per-sample
# attention mask.  60 training signals per game per forward pass.
# Usage:  sbatch train_gpt_shuffled_v3.sh
# Overrides: EPOCHS=20 BATCH_SIZE=1024 MAX_GAME_LEN=60 LOAD_CKPT=path

#SBATCH --job-name=gpt_shuf_v3
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=160GB
#SBATCH --gres=gpu:8
#SBATCH --output=logs/train_gpt_shuffled_v3_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Train GPT on shuffled context with 60 query positions"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "EPOCHS=${EPOCHS:-20}  BATCH_SIZE=${BATCH_SIZE:-1024}  MAX_GAME_LEN=${MAX_GAME_LEN:-60}"
echo "Started at: $(date)"
echo "============================================"

# Batch reduced to 1024 (= 128/GPU) because block_size doubled and masks
# add ~14KB/sample on top of inputs.  Tune up if memory headroom allows.
EPOCHS=${EPOCHS:-20} BATCH_SIZE=${BATCH_SIZE:-1024} \
    NUM_WORKERS=${NUM_WORKERS:-16} MAX_GAME_LEN=${MAX_GAME_LEN:-60} \
    LOAD_CKPT=${LOAD_CKPT:-} \
    python train_gpt_shuffled_v3.py

echo "Completed at: $(date)"
