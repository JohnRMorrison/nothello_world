#!/bin/bash
# ============================================================================
# SLURM job: endgame decision-tree MLP at scale.
#
#   sbatch train_endgame_tree.sh                       # defaults
#   sbatch train_endgame_tree.sh 20000 5000 15 5       # ngames_tr ngames_te depth min_leaf
# ============================================================================

#SBATCH --job-name=endgame_tree
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/endgame_tree_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs ckpts_endgame

cd $SLURM_SUBMIT_DIR

NUM_TRAIN=${1:-20000}
NUM_TEST=${2:-5000}
MAX_DEPTH=${3:-15}
MIN_LEAF=${4:-50}     # bumped again; endgame trees grow bigger than opening
N_JOBS=${5:-8}        # reduced from 16 to lower peak RAM during parallel fit

echo "============================================"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              $(hostname)"
echo "Started at:        $(date)"
echo "num_train_games:   ${NUM_TRAIN}"
echo "num_test_games:    ${NUM_TEST}"
echo "tree_max_depth:    ${MAX_DEPTH}"
echo "min_samples_leaf:  ${MIN_LEAF}"
echo "============================================"

OUT="ckpts_endgame/endgame_tree_g${NUM_TRAIN}_d${MAX_DEPTH}_ml${MIN_LEAF}.pt"

CUDA_VISIBLE_DEVICES=0 python endgame_tree_mlp.py \
    --num-train-games ${NUM_TRAIN} \
    --num-test-games ${NUM_TEST} \
    --endgame-ply 10 \
    --tree-max-depth ${MAX_DEPTH} \
    --tree-min-samples-leaf ${MIN_LEAF} \
    --tree-n-jobs ${N_JOBS} \
    --probe-epochs 30 \
    --device cuda \
    --out ${OUT}

echo "Completed at: $(date)"
