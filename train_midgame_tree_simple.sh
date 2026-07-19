#!/bin/bash
# ============================================================================
# SLURM job: midgame tree MLP with SIMPLIFIED input featurization.
#
# Input becomes played + even + recent (60x3 + mover_parity) — no separate
# order-node hidden bank, no movegrid, no --use-move-grid.  Trees fit on the
# enlarged binary input and can produce moveset × recency conjunctions
# directly as tree paths.
#
# Variants:
#   simple_K5    --input-recent-Ks 5           input 60x3 = 181-d
#   simple_K10   --input-recent-Ks 10          input 60x3 = 181-d
#   simple_multi --input-recent-Ks 1,2,5,10,20 input 60x7 = 421-d
#
# Everything else matches the k1_variants baseline: ReLU + top-K=50 tree
# paths + 5-seed probe ensemble + 100 epochs.  The base line (no --input-
# recent-Ks) equals train_midgame_tree_k1_variants.sh base for direct
# comparison.
# ============================================================================

#SBATCH --job-name=midgame_smpl
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=240GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/midgame_smpl_%j.out
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
        RECENT_KS="5"
        TAG="k5"
        ;;
    simple_K10)
        RECENT_KS="10"
        TAG="k10"
        ;;
    simple_multi)
        RECENT_KS="1,2,5,10,20"
        TAG="multi"
        ;;
    bank_K5)
        # Recent bits piped directly to hidden layer, not tree input.
        BANK_KS="5"
        TAG="bank_k5"
        ;;
    bank_multi)
        BANK_KS="1,2,5,10,20"
        TAG="bank_multi"
        ;;
    bank_multi_probor)
        # bank_multi with the noisy-OR state readout instead of linear.
        BANK_KS="1,2,5,10,20"
        TAG="bank_multi_probor"
        STATE_READOUT="probor"
        ;;
    *)
        echo "unknown VARIANT '${VARIANT}' — use: simple_K5 simple_K10 simple_multi bank_K5 bank_multi bank_multi_probor"
        exit 1
        ;;
esac

STATE_READOUT=${STATE_READOUT:-linear}

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "variant:           ${VARIANT}  (${TAG})"
echo "recent Ks:         ${RECENT_KS}"
echo "num_train_games:   ${NUM_TRAIN}"
echo "num_test_games:    ${NUM_TEST}"
echo "tree_max_depth:    ${MAX_DEPTH}"
echo "min_samples_leaf:  ${MIN_LEAF}"
echo "ply range:         [${PLY_MIN}, ${PLY_MAX})"
echo "top_k_per_cell:    ${TOP_K}"
echo "============================================"

# Cache filename includes the recent-Ks tag so different variants don't clash.
# bank_K5 shares a cache with simple_K5 (same Xnp — both compute recent-K bits
# at sample time; the flag only differs in how they're consumed downstream).
if [ -n "${RECENT_KS:-}" ]; then
    RECENT_FLAG="--input-recent-Ks ${RECENT_KS}"
    CACHE_TAG=$(echo "${RECENT_KS}" | tr ',' '_')
elif [ -n "${BANK_KS:-}" ]; then
    RECENT_FLAG="--recent-Ks-as-hidden ${BANK_KS}"
    CACHE_TAG=$(echo "${BANK_KS}" | tr ',' '_')
else
    RECENT_FLAG=""
    CACHE_TAG="none"
fi
OUT="ckpts_midgame/midgame_smpl_${TAG}_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}_p${PLY_MIN}-${PLY_MAX}.pt"
CACHE_TR="ckpts_midgame/cache/midgame_g${NUM_TRAIN}_p${PLY_MIN}-${PLY_MAX}_rK${CACHE_TAG}_tr.npz"
CACHE_TE="ckpts_midgame/cache/midgame_g${NUM_TEST}_p${PLY_MIN}-${PLY_MAX}_rK${CACHE_TAG}_te.npz"

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
    --state-readout ${STATE_READOUT} \
    ${RECENT_FLAG} \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    --out ${OUT}

echo "Completed at: $(date)"
