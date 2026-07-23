#!/bin/bash
# ============================================================================
# SLURM job: streaming STATE-decoding probe on tree-leaf activations.
#
# Linear board-state decoder on the hidden layer of a pattern-tree
# checkpoint (analogous to Nanda's layer-6 probe).  Streams chunk_ext
# files one at a time (constant memory).  Per-chunk resume: if timed out,
# resubmit the SAME command and it continues from the last checkpoint.
#
# Required env vars:
#   LOAD_TREES=path/to/tree_checkpoint.pt
#
# Optional env vars:
#   CHUNK_DIR   (default: heuristic_probe_results/feature_chunks)
#   NUM_GAMES   (default 6000000)
#   EPOCHS      (default 1)
#   CANONICALIZE_MOVER=1  (recommended for canonical tree checkpoints)
#   USE_RELU=1  (default: step activation)
#   CHECKPOINT_EVERY (default 5)
# ============================================================================

#SBATCH --job-name=stream_state
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/stream_state_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame ckpts_midgame/resume

cd $SLURM_SUBMIT_DIR

LOAD_TREES=${LOAD_TREES:?Must set LOAD_TREES to a tree checkpoint path}
CHUNK_DIR=${CHUNK_DIR:-experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks}
NUM_GAMES=${NUM_GAMES:-6000000}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-1e-3}
PLY_MIN=${PLY_MIN:-10}
PLY_MAX=${PLY_MAX:-50}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
RELU_FLAG=""
if [ -n "${USE_RELU:-}" ]; then
    RELU_FLAG="--use-relu"
fi
CANONICALIZE_FLAG=""
if [ -n "${CANONICALIZE_MOVER:-}" ]; then
    CANONICALIZE_FLAG="--canonicalize-mover"
fi
MAX_POS_FLAG=""
if [ -n "${MAX_POSITIONS_PER_FILE:-}" ]; then
    MAX_POS_FLAG="--max-positions-per-file ${MAX_POSITIONS_PER_FILE}"
fi

TS=$(date +%Y%m%d_%H%M%S)
TREE_TAG=$(basename "${LOAD_TREES}" .pt)
OUT="ckpts_midgame/streamstate_g${NUM_GAMES}_ep${EPOCHS}_${TREE_TAG}_j${SLURM_JOB_ID}_${TS}.pt"
# Stable resume sidecar keyed by run config (NOT job id / timestamp) so a
# resubmitted job still finds and continues the previous run's checkpoint.
RESUME_STATE="ckpts_midgame/resume/streamstate_g${NUM_GAMES}_ep${EPOCHS}_${TREE_TAG}.resume"

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "LOAD_TREES:        ${LOAD_TREES}"
echo "CHUNK_DIR:         ${CHUNK_DIR}"
echo "NUM_GAMES:         ${NUM_GAMES}"
echo "EPOCHS:            ${EPOCHS}"
echo "PLY_RANGE:         [${PLY_MIN}, ${PLY_MAX})"
echo "canonicalize:      ${CANONICALIZE_MOVER:-<off>}"
echo "use_relu:          ${USE_RELU:-<step>}"
echo "OUT:               ${OUT}"
echo "RESUME_STATE:      ${RESUME_STATE}"
echo "CHECKPOINT_EVERY:  ${CHECKPOINT_EVERY}"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_streaming_state.py \
    --load-trees-from ${LOAD_TREES} \
    --chunk-dir ${CHUNK_DIR} \
    --num-train-games ${NUM_GAMES} \
    --ply-min ${PLY_MIN} \
    --ply-max ${PLY_MAX} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --checkpoint-every ${CHECKPOINT_EVERY} \
    --resume --resume-from ${RESUME_STATE} \
    ${RELU_FLAG} \
    ${CANONICALIZE_FLAG} \
    ${MAX_POS_FLAG} \
    --out ${OUT}

echo "Completed at: $(date)"
