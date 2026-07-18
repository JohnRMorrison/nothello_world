#!/bin/bash
# ============================================================================
# SLURM job: midgame tree MLP + compact K=1 count-node bank on top of ReLU.
#
# Compares three compact banks (Options 2/3/4 from planning) meant to add
# count-magnitude signal without drowning the probe in extra units.
#
#   sbatch train_midgame_tree_k1_variants.sh nbhd    # Option 2: 60 nbhd nodes
#   sbatch train_midgame_tree_k1_variants.sh rays    # Option 4: 42 line rays
#   sbatch train_midgame_tree_k1_variants.sh boost   # Option 3: 60 boosted
#   sbatch train_midgame_tree_k1_variants.sh base    # baseline: no count nodes
#
# All variants share: ReLU activation, top-K=50 tree paths, 5 probe seeds,
# 100 probe epochs, ply range [10, 50).
# ============================================================================

#SBATCH --job-name=midgame_k1var
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=240GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/midgame_k1var_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame ckpts_midgame/cache

cd $SLURM_SUBMIT_DIR

VARIANT=${1:-base}       # nbhd | rays | boost | base
NUM_TRAIN=${2:-20000}
NUM_TEST=${3:-5000}
MAX_DEPTH=${4:-15}
MIN_LEAF=${5:-50}
PLY_MIN=${6:-10}
PLY_MAX=${7:-50}
TOP_K=${8:-50}

case "${VARIANT}" in
    nbhd)
        VARIANT_FLAGS="--include-neighborhood-count-nodes"
        TAG="nbhd"
        ;;
    rays)
        VARIANT_FLAGS="--include-ray-count-nodes"
        TAG="rays"
        ;;
    boost)
        # Option 3: boost 60 candidates from the 1824-node structured pool
        # (no random pool → --boost-candidate-pool 0).
        VARIANT_FLAGS="--boost-count-rounds 1 --boost-count-per-round 60 \
                        --boost-candidate-pool 0"
        TAG="boost60"
        ;;
    base)
        VARIANT_FLAGS=""
        TAG="base"
        ;;
    *)
        echo "unknown VARIANT '${VARIANT}' — use one of: nbhd rays boost base"
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

OUT="ckpts_midgame/midgame_k1var_${TAG}_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}_p${PLY_MIN}-${PLY_MAX}.pt"
CACHE_TR="ckpts_midgame/cache/midgame_g${NUM_TRAIN}_p${PLY_MIN}-${PLY_MAX}_tr.npz"
CACHE_TE="ckpts_midgame/cache/midgame_g${NUM_TEST}_p${PLY_MIN}-${PLY_MAX}_te.npz"

CUDA_VISIBLE_DEVICES=0 python midgame_tree_mlp.py \
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
    ${VARIANT_FLAGS} \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    --out ${OUT}

echo "Completed at: $(date)"
