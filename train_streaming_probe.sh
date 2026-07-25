#!/bin/bash
# ============================================================================
# SLURM job: streaming probe training on pre-generated pickle files.
#
# Loads trees from a saved checkpoint (LOAD_TREES env), streams through
# pickle files chunk-by-chunk (~100K games each), and trains a legal-move
# probe on the enlarged data.  Fits in memory even at 60M+ games because
# only ONE chunk's hidden layer is in GPU at a time.
#
# Required env vars:
#   LOAD_TREES=path/to/checkpoint.pt
#
# Optional env vars:
#   PICKLE_DIR=data/othello_synthetic (default)
#   NUM_GAMES=6000000
#   PROBE_TYPE=linpo | strupo  (default linpo)
#   EPOCHS=3
#   USE_RELU=1 (default: step activation, bool memory)
# ============================================================================

#SBATCH --job-name=stream_probe
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/stream_probe_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame

cd $SLURM_SUBMIT_DIR

# FLANKING_ONLY=1 → no tree bank; hidden layer is the 960 flanking patterns
# alone (diagnostic).  LOAD_TREES is then optional and ignored.
FLANKING_ONLY=${FLANKING_ONLY:-}
# NO_FLANKING=1 → drop the 960 flanking patterns; TREE-ONLY hidden layer.
NO_FLANKING=${NO_FLANKING:-}
FLANK_FLAG=""
NOFLANK_TAG=""
if [ -n "${FLANKING_ONLY}" ]; then
    FLANK_FLAG="--flanking-only"
    LOAD_TREES=${LOAD_TREES:-flankingonly}
else
    LOAD_TREES=${LOAD_TREES:?Must set LOAD_TREES to a checkpoint path}
    if [ -n "${NO_FLANKING}" ]; then
        FLANK_FLAG="--no-flanking"
        NOFLANK_TAG="_noflank"       # keep OUT/RESUME distinct from with-flanking
    fi
fi
DATA_SOURCE=${DATA_SOURCE:-chunk-ext}
PICKLE_DIR=${PICKLE_DIR:-data/othello_synthetic}
CHUNK_DIR=${CHUNK_DIR:-experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks}
NUM_GAMES=${NUM_GAMES:-6000000}
NUM_TEST_GAMES=${NUM_TEST_GAMES:-100000}
PROBE_TYPE=${PROBE_TYPE:-linpo}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-1e-3}
PLY_MIN=${PLY_MIN:-0}
PLY_MAX=${PLY_MAX:-60}
RECENT_KS=${RECENT_KS-1,2,5,10,20}
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
# Include the LOAD_TREES basename so parallel jobs with different tree
# checkpoints don't collide when they share a start-second.  Also
# include the SLURM job id as a tiebreaker.
TREE_TAG=$(basename "${LOAD_TREES}" .pt)${NOFLANK_TAG}
OUT="ckpts_midgame/stream_${PROBE_TYPE}_g${NUM_GAMES}_ep${EPOCHS}_${TREE_TAG}_j${SLURM_JOB_ID}_${TS}.pt"

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "LOAD_TREES:        ${LOAD_TREES}"
echo "DATA_SOURCE:       ${DATA_SOURCE}"
echo "PICKLE_DIR:        ${PICKLE_DIR}"
echo "CHUNK_DIR:         ${CHUNK_DIR}"
echo "NUM_GAMES:         ${NUM_GAMES}"
echo "PROBE_TYPE:        ${PROBE_TYPE}"
echo "EPOCHS:            ${EPOCHS}"
echo "BATCH_SIZE:        ${BATCH_SIZE}"
echo "PLY_RANGE:         [${PLY_MIN}, ${PLY_MAX})"
echo "RECENT_KS:         ${RECENT_KS}"
echo "canonicalize:      ${CANONICALIZE_MOVER:-<off>}"
echo "use_relu:          ${USE_RELU:-<step>}"
# Stable resume sidecar keyed by run config (NOT job id / timestamp) so a
# resubmitted job — which gets a new OUT name — still finds and continues
# from the previous run's checkpoint.  Removed automatically on completion.
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
N_SEEDS=${N_SEEDS:-1}
# Multi-seed ensemble: tag OUT/resume so an N-seed run doesn't collide with the
# single-seed one.  Single-seed naming is unchanged (SEED_TAG empty).
SEED_TAG=""
if [ "${N_SEEDS}" != "1" ]; then SEED_TAG="_s${N_SEEDS}"; OUT="${OUT%.pt}${SEED_TAG}.pt"; fi
RESUME_STATE="ckpts_midgame/resume/stream_${PROBE_TYPE}_g${NUM_GAMES}_ep${EPOCHS}_${TREE_TAG}${SEED_TAG}.resume"
mkdir -p ckpts_midgame/resume
echo "N_SEEDS:           ${N_SEEDS}"
echo "OUT:               ${OUT}"
echo "RESUME_STATE:      ${RESUME_STATE}"
echo "CHECKPOINT_EVERY:  ${CHECKPOINT_EVERY}"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u train_streaming_probe.py \
    --load-trees-from ${LOAD_TREES} \
    ${FLANK_FLAG} \
    --data-source ${DATA_SOURCE} \
    --pickle-dir ${PICKLE_DIR} \
    --chunk-dir ${CHUNK_DIR} \
    --num-train-games ${NUM_GAMES} \
    --num-test-games ${NUM_TEST_GAMES} \
    --recent-Ks "${RECENT_KS}" \
    --probe-type ${PROBE_TYPE} \
    --ply-min ${PLY_MIN} \
    --ply-max ${PLY_MAX} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --n-seeds ${N_SEEDS} \
    --checkpoint-every ${CHECKPOINT_EVERY} \
    --resume --resume-from ${RESUME_STATE} \
    ${RELU_FLAG} \
    ${CANONICALIZE_FLAG} \
    ${MAX_POS_FLAG} \
    --out ${OUT}

echo "Completed at: $(date)"
