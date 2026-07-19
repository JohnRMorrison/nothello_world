#!/bin/bash
# ============================================================================
# SLURM job: midgame tree MLP + one order-aware hidden bank on top.
#
# Trees fit on played_even (121-d, unchanged), one order-aware bank appended
# to the hidden layer as extra units.  Runs on top of ReLU activation + top-K
# tree paths + 5-seed probe ensemble, matching train_midgame_tree_k1_variants.sh
# so results are directly comparable.
#
#   sbatch train_midgame_tree_order.sh turnbucket
#   sbatch train_midgame_tree_order.sh recency
#   sbatch train_midgame_tree_order.sh ordinal
#   sbatch train_midgame_tree_order.sh pairwise
#   sbatch train_midgame_tree_order.sh streak
#   sbatch train_midgame_tree_order.sh all       # all five at once
# ============================================================================

#SBATCH --job-name=midgame_ord
#SBATCH -c 16
#SBATCH --time=8:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/midgame_ord_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame ckpts_midgame/cache

cd $SLURM_SUBMIT_DIR

VARIANT=${1:-turnbucket}
NUM_TRAIN=${2:-20000}
NUM_TEST=${3:-5000}
MAX_DEPTH=${4:-15}
MIN_LEAF=${5:-50}
PLY_MIN=${6:-10}
PLY_MAX=${7:-50}
TOP_K=${8:-50}

case "${VARIANT}" in
    turnbucket)
        VARIANT_FLAGS="--include-turn-bucket-nodes --turn-bucket-size 10"
        TAG="tb"
        ;;
    recency)
        # Now with parity variants (60 × 5 × 3 = 900 units).
        VARIANT_FLAGS="--include-recency-nodes --recency-Ks 1,2,5,10,20"
        TAG="rec"
        ;;
    recency_wide)
        # Wider K sweep (60 × 7 × 3 = 1260 units).
        VARIANT_FLAGS="--include-recency-nodes --recency-Ks 1,2,5,10,20,30,40"
        TAG="rec_wide"
        ;;
    recency_only)
        # Ablation: recency alone, no tree paths.
        VARIANT_FLAGS="--include-recency-nodes --recency-Ks 1,2,5,10,20,30,40 --skip-tree-fit"
        TAG="rec_only"
        ;;
    ordinal)
        VARIANT_FLAGS="--include-ordinal-nodes"
        TAG="ord"
        ;;
    pairwise)
        VARIANT_FLAGS="--include-pairwise-order-nodes --pairwise-max-chebyshev 2"
        TAG="pw"
        ;;
    streak)
        VARIANT_FLAGS="--include-streak-nodes --pairwise-max-chebyshev 2 --streak-N-gap 3"
        TAG="strk"
        ;;
    all)
        VARIANT_FLAGS="--include-turn-bucket-nodes --turn-bucket-size 10 \
                        --include-recency-nodes --recency-Ks 1,2,5,10,20 \
                        --include-ordinal-nodes \
                        --include-pairwise-order-nodes --pairwise-max-chebyshev 2 \
                        --include-streak-nodes --streak-N-gap 3"
        TAG="all5"
        ;;
    *)
        echo "unknown VARIANT '${VARIANT}' — use: turnbucket recency ordinal pairwise streak all"
        exit 1
        ;;
esac

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "variant:           ${VARIANT}  (${TAG})"
echo "variant_flags:     ${VARIANT_FLAGS}"
echo "num_train_games:   ${NUM_TRAIN}"
echo "num_test_games:    ${NUM_TEST}"
echo "tree_max_depth:    ${MAX_DEPTH}"
echo "min_samples_leaf:  ${MIN_LEAF}"
echo "ply range:         [${PLY_MIN}, ${PLY_MAX})"
echo "top_k_per_cell:    ${TOP_K}"
echo "============================================"

# NOTE: movegrid-inclusive cache is DIFFERENT from the playedeven-only cache
# used by train_midgame_tree_k1_variants.sh.  Order-node runs share one MG
# cache per (games, ply-range).
OUT="ckpts_midgame/midgame_ord_${TAG}_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}_p${PLY_MIN}-${PLY_MAX}.pt"
CACHE_TR="ckpts_midgame/cache/midgame_g${NUM_TRAIN}_p${PLY_MIN}-${PLY_MAX}_mg_tr.npz"
CACHE_TE="ckpts_midgame/cache/midgame_g${NUM_TEST}_p${PLY_MIN}-${PLY_MAX}_mg_te.npz"

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
    --use-move-grid \
    ${VARIANT_FLAGS} \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    --out ${OUT}

echo "Completed at: $(date)"
