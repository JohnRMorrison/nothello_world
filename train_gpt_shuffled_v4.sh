#!/bin/bash
# v4: cell-indexed context (no shuffle) + 60 query positions via per-sample
# attention mask.  Direct MLP analog: position c in the sequence corresponds
# to cell c, just like the MLP's played+even feature vector.
# Usage:  sbatch train_gpt_shuffled_v4.sh
# Overrides: EPOCHS=20 BATCH_SIZE=1024 LOAD_CKPT=path

#SBATCH --job-name=gpt_shuf_v4
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=160GB
#SBATCH --gres=gpu:8
#SBATCH --output=logs/train_gpt_shuffled_v4_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Train GPT with cell-indexed context (MLP analog)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "EPOCHS=${EPOCHS:-20}  BATCH_SIZE=${BATCH_SIZE:-1024}"
echo "Started at: $(date)"
echo "============================================"

# If LOAD_CKPT isn't explicitly set, pick up the most recent v4 ckpt so
# this script can be safely re-run as part of a dependency chain.
if [ -z "${LOAD_CKPT:-}" ]; then
    LATEST_CKPT=$(ls -t ckpts/gpt_shuffled_v4_*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Auto-resuming from latest checkpoint: $LATEST_CKPT"
        LOAD_CKPT="$LATEST_CKPT"
    fi
fi

EPOCHS=${EPOCHS:-20} BATCH_SIZE=${BATCH_SIZE:-1024} \
    NUM_WORKERS=${NUM_WORKERS:-16} \
    LOAD_CKPT="${LOAD_CKPT:-}" \
    CONSTANT_LR="${CONSTANT_LR:-}" \
    python train_gpt_shuffled_v4.py

echo "Completed at: $(date)"
