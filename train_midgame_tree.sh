#!/bin/bash
# ============================================================================
# SLURM job: midgame decision-tree MLP.
#
#   sbatch train_midgame_tree.sh                              # defaults
#   sbatch train_midgame_tree.sh 20000 5000 15 5 10 50 nostab
#     ngames_tr ngames_te depth min_leaf ply_min ply_max stability_flag
#
# stability_flag: 'stab' → add stability feature bank, otherwise omit.
# ============================================================================

#SBATCH --job-name=midgame_tree
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=240GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/midgame_tree_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_midgame

cd $SLURM_SUBMIT_DIR

NUM_TRAIN=${1:-20000}
NUM_TEST=${2:-5000}
MAX_DEPTH=${3:-15}
MIN_LEAF=${4:-50}     # bumped from 5 — 800k midgame positions blow RAM otherwise
PLY_MIN=${5:-10}
PLY_MAX=${6:-50}
STAB=${7:-nostab}
N_JOBS=${8:-1}        # single-threaded tree fit — no worker copies, safest

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "num_train_games:   ${NUM_TRAIN}"
echo "num_test_games:    ${NUM_TEST}"
echo "tree_max_depth:    ${MAX_DEPTH}"
echo "min_samples_leaf:  ${MIN_LEAF}"
echo "ply range:         [${PLY_MIN}, ${PLY_MAX})"
echo "stability:         ${STAB}"
echo "============================================"

if [ "${STAB}" = "stab" ]; then
    STAB_FLAG="--add-stability-features"
    STAB_TAG="_stab"
else
    STAB_FLAG=""
    STAB_TAG=""
fi

OUT="ckpts_midgame/midgame_tree_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}_p${PLY_MIN}-${PLY_MAX}${STAB_TAG}.pt"

# Cache the sampled positions so reruns skip the ~15-min sampling step.
mkdir -p ckpts_midgame/cache
CACHE_TR="ckpts_midgame/cache/midgame_g${NUM_TRAIN}_p${PLY_MIN}-${PLY_MAX}_tr.npz"
CACHE_TE="ckpts_midgame/cache/midgame_g${NUM_TEST}_p${PLY_MIN}-${PLY_MAX}_te.npz"

CUDA_VISIBLE_DEVICES=0 python midgame_tree_mlp.py \
    ${STAB_FLAG} \
    --num-train-games ${NUM_TRAIN} \
    --num-test-games ${NUM_TEST} \
    --ply-min ${PLY_MIN} \
    --ply-max ${PLY_MAX} \
    --tree-max-depth ${MAX_DEPTH} \
    --tree-min-samples-leaf ${MIN_LEAF} \
    --tree-n-jobs ${N_JOBS} \
    --probe-epochs 30 \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    --out ${OUT}

echo "Completed at: $(date)"
