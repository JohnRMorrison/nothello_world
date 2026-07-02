#!/bin/bash
# Train a transformer aggregator over ensemble MLP predictions.
#
# Env vars:
#   MULTI_CKPTS   : space-sep list of multi-seed checkpoint paths
#   TRAIN_CHUNK_START (default 20)
#   NUM_TRAIN_CHUNKS  (default 2)   ~500K games each
#   EVAL_CHUNK    (default 39)
#   EPOCHS        (default 3)
#   BATCH_SIZE    (default 1024)
#   D_MODEL       (default 64)
#   N_HEADS       (default 4)
#   N_LAYERS      (default 2)
#   SAVE_PATH     (default derived)

#SBATCH --job-name=mlp_token_tf
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_token_tf_%j.out
#SBATCH --account=nklab
#SBATCH --partition=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

MULTI_CKPTS=${MULTI_CKPTS:-}
TRAIN_CHUNK_START=${TRAIN_CHUNK_START:-20}
NUM_TRAIN_CHUNKS=${NUM_TRAIN_CHUNKS:-2}
EVAL_CHUNK=${EVAL_CHUNK:-39}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1024}
D_MODEL=${D_MODEL:-64}
N_HEADS=${N_HEADS:-4}
N_LAYERS=${N_LAYERS:-2}
SAVE_PATH=${SAVE_PATH:-}

if [ -z "$MULTI_CKPTS" ]; then
    echo "ERROR: set MULTI_CKPTS to a space-separated list of ckpt paths"
    exit 1
fi

echo "============================================"
echo "MLP-token transformer training"
echo "  ckpts: $MULTI_CKPTS"
echo "  train chunks [$TRAIN_CHUNK_START:$((TRAIN_CHUNK_START + NUM_TRAIN_CHUNKS - 1))]"
echo "  eval chunk: $EVAL_CHUNK"
echo "  epochs=$EPOCHS batch=$BATCH_SIZE"
echo "  d_model=$D_MODEL n_heads=$N_HEADS n_layers=$N_LAYERS"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

SAVE_ARG=""
if [ -n "$SAVE_PATH" ]; then
    SAVE_ARG="--save-path $SAVE_PATH"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_mlp_token_transformer.py \
    --multi-ckpts $MULTI_CKPTS \
    --train-chunk-start $TRAIN_CHUNK_START \
    --num-train-chunks $NUM_TRAIN_CHUNKS \
    --eval-chunk $EVAL_CHUNK \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --n-layers $N_LAYERS \
    $SAVE_ARG

echo "Completed at: $(date)"
