#!/bin/bash
# ============================================================================
# SLURM job: midgame tree MLP with --task both (state + legal-move probes).
#
# Runs the SAME state pipeline as train_midgame_tree_simple.sh, plus three
# legal-move predictors on the same hidden layer:
#   - BCE:     linear(H, 64) + sigmoid + BCE loss.
#   - probOR:  noisy-OR head — 1 - exp(-Σ w_ic h_i) — with BCE loss.
#   - derived: Option C — apply flanking rule to state predictions.
#
# Uses a DIFFERENT cache (_L suffix) because sample also collects the
# per-position legal-move mask.
#
# Variants pick the input featurization (identical to simple_*):
#   sbatch train_midgame_tree_legal.sh simple_K5
#   sbatch train_midgame_tree_legal.sh simple_K10
#   sbatch train_midgame_tree_legal.sh simple_multi
#   sbatch train_midgame_tree_legal.sh base           # no recent input
# ============================================================================

#SBATCH --job-name=midgame_leg
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=240GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/midgame_leg_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame ckpts_midgame/cache

cd $SLURM_SUBMIT_DIR

VARIANT=${1:-simple_K5}
NUM_TRAIN=${2:-20000}
NUM_TEST=${3:-5000}
MAX_DEPTH=${4:-15}
MIN_LEAF=${5:-50}
PLY_MIN=${6:-10}
PLY_MAX=${7:-50}
TOP_K=${8:-50}

case "${VARIANT}" in
    simple_K5)
        RECENT_ARG="--input-recent-Ks 5"
        TAG="k5"
        ;;
    simple_K10)
        RECENT_ARG="--input-recent-Ks 10"
        TAG="k10"
        ;;
    simple_multi)
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20"
        TAG="multi"
        ;;
    base)
        RECENT_ARG=""
        TAG="base"
        ;;
    bank_K5)
        RECENT_ARG="--recent-Ks-as-hidden 5"
        TAG="bank_k5"
        ;;
    bank_multi)
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20"
        TAG="bank_multi"
        ;;
    bank_multi_legaltrees)
        # bank_multi input featurization + trees fit for legality
        # (Option 1: --tree-target legal).  State probe is skipped.
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20"
        TAG="bank_multi_legaltrees"
        TREE_TARGET="legal"
        ;;
    bank_multi_flanking)
        # bank_multi + 960 hand-crafted flanking patterns as extra hidden
        # units (Option 3): trees for state, recent bits, 960 patterns.
        # Legal probes read the enlarged hidden layer.
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt"
        TAG="bank_multi_flanking"
        ;;
    bank_multi_flanking_legaltrees)
        # Both: trees fit for legal + 960 flanking-pattern hidden units.
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt"
        TAG="bank_multi_flanking_legaltrees"
        TREE_TARGET="legal"
        ;;
    *)
        echo "unknown VARIANT '${VARIANT}' — use: simple_K5 simple_K10 simple_multi bank_K5 bank_multi base"
        exit 1
        ;;
esac

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "variant:           ${VARIANT}  (${TAG})"
echo "input flag:        ${RECENT_ARG}"
echo "num_train_games:   ${NUM_TRAIN}"
echo "num_test_games:    ${NUM_TEST}"
echo "tree_max_depth:    ${MAX_DEPTH}"
echo "min_samples_leaf:  ${MIN_LEAF}"
echo "ply range:         [${PLY_MIN}, ${PLY_MAX})"
echo "top_k_per_cell:    ${TOP_K}"
echo "task:              both"
echo "============================================"

TREE_TARGET=${TREE_TARGET:-state}

# Cache path includes _L suffix so it does not clash with state-only caches
# that have 3-tuple contents.  The 4th cached array is the legal-move mask.
OUT="ckpts_midgame/midgame_leg_${TAG}_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}_p${PLY_MIN}-${PLY_MAX}.pt"
# bank_multi_legaltrees shares cache with bank_multi (same Xnp sample).
CACHE_TAG=${TAG}
case "${TAG}" in
    bank_multi_legaltrees|bank_multi_flanking|bank_multi_flanking_legaltrees)
        CACHE_TAG="bank_multi"
        ;;
esac
CACHE_TR="ckpts_midgame/cache/midgame_g${NUM_TRAIN}_p${PLY_MIN}-${PLY_MAX}_r${CACHE_TAG}_L_tr.npz"
CACHE_TE="ckpts_midgame/cache/midgame_g${NUM_TEST}_p${PLY_MIN}-${PLY_MAX}_r${CACHE_TAG}_L_te.npz"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u midgame_tree_mlp.py \
    --num-train-games ${NUM_TRAIN} \
    --num-test-games ${NUM_TEST} \
    --ply-min ${PLY_MIN} \
    --ply-max ${PLY_MAX} \
    --tree-max-depth ${MAX_DEPTH} \
    --tree-min-samples-leaf ${MIN_LEAF} \
    --tree-n-jobs 1 \
    --top-k-per-cell ${TOP_K} \
    --hidden-activation relu \
    --probe-epochs 100 \
    --probe-seeds 5 \
    --task both \
    --tree-target ${TREE_TARGET} \
    --legal-modes bce,probor,derived \
    --legal-probe-epochs 100 \
    ${RECENT_ARG} \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    --out ${OUT}

echo "Completed at: $(date)"
