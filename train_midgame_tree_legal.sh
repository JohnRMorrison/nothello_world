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
    pattern_trees_strupo)
        # 960 per-pattern trees + StruPO probe (per-pattern linear over
        # leaves -> sigmoid -> prob-OR per cell).  Assumes trees are
        # loaded from a prior pattern_trees checkpoint (LOAD_TREES env),
        # so tree fit is skipped entirely.  Ideal for iterating the
        # linear weights on a larger training set (100K games).
        RECENT_ARG="--include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1 --skip-state-probe"
        if [ -n "${LOAD_TREES:-}" ]; then
            RECENT_ARG="${RECENT_ARG} --load-trees-from ${LOAD_TREES}"
        fi
        TAG="pattern_trees_strupo"
        LEGAL_MODES_OVERRIDE="patterns_structured_probor"
        ;;
    bank_multi_flanking_linpo)
        # bank_multi_flanking but only trains the LinPO legal probe.
        # Skips the state probe entirely + all other legal predictors,
        # AND reuses trees from a saved bank_multi_flanking checkpoint
        # (LOAD_TREES env var) so we don't wait ~40 min for tree refit.
        # Usage:
        #   LOAD_TREES=ckpts_midgame/midgame_leg_bank_multi_flanking_g20000_d15_ml50_p10-50.pt \
        #     sbatch train_midgame_tree_legal.sh bank_multi_flanking_linpo 100000 5000 ...
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --skip-state-probe"
        if [ -n "${LOAD_TREES:-}" ]; then
            RECENT_ARG="${RECENT_ARG} --load-trees-from ${LOAD_TREES}"
        fi
        TAG="bank_multi_flanking_linpo"
        LEGAL_MODES_OVERRIDE="patterns_linear_probor"
        ;;
    bank_multi_flanking_legaltrees)
        # Both: trees fit for legal + 960 flanking-pattern hidden units.
        RECENT_ARG="--recent-Ks-as-hidden 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt"
        TAG="bank_multi_flanking_legaltrees"
        TREE_TARGET="legal"
        ;;
    pattern_trees)
        # 960 per-pattern trees (single tree per pattern).  Trees fit on
        # played+even+mover_parity+recent bits.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1"
        TAG="pattern_trees"
        ;;
    pattern_trees_bag10)
        # 960 patterns × 10 bagged trees each = 9600 trees.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 10"
        TAG="pattern_trees_bag10"
        ;;
    pattern_trees_unbalanced)
        # 960 per-pattern trees with class_weight=none — better calibration
        # under prob-OR combination.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1 --pattern-class-weight none"
        TAG="pattern_trees_unbalanced"
        ;;
    pattern_trees_bag10_unbalanced)
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 10 --pattern-class-weight none"
        TAG="pattern_trees_bag10_unbalanced"
        ;;
    pattern_trees_depth20)
        # Deeper pattern trees; may discover finer flanking conjunctions.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1"
        TAG="pattern_trees_depth20"
        # depth override via the 4th positional arg (MAX_DEPTH).  User can
        # invoke as: sbatch ... pattern_trees_depth20 20000 5000 20 50 ...
        ;;
    pattern_trees_leaf10)
        # Looser pruning; leaves must have >= 10 samples rather than 50.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1"
        TAG="pattern_trees_leaf10"
        ;;
    pattern_trees_rf)
        # RandomForestClassifier per pattern (10 estimators with feature
        # subsampling), instead of hand-bagged plain trees.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 10 --pattern-use-random-forest"
        TAG="pattern_trees_rf"
        ;;
    pattern_trees_no_recent)
        # Pure moveset: played + even + mover_parity only (121-d).  No
        # recency bits in the tree input.  Cleanest paper baseline.
        RECENT_ARG="--include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1"
        TAG="pattern_trees_no_recent"
        ;;
    pattern_trees_no_recent_canonical)
        # Mine/yours moveset: replaces `even` with `placed_as_mover` so
        # trees see mover-relative color directly, no need to condition
        # on mover_parity internally.  Cleanest structure for flanking
        # patterns (which are mover-relative by definition).
        RECENT_ARG="--include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1 --canonicalize-mover"
        TAG="pattern_trees_no_recent_canonical"
        ;;
    bank_multi_flanking_legaltrees_patterntrees)
        # Combined bank: legal-target trees (3200 leaves) + pattern-target
        # trees (48000 leaves) + hand-crafted flanking pattern activations
        # (960).  Two-stage: fit legal trees first, save checkpoint; then
        # RE-run with --tree-target patterns AND --load-trees-from that
        # checkpoint to combine.  Requires new tree-target=both mode; for
        # now, easier to run pattern_trees + StruPO and separately eval
        # bank_multi_legaltrees + BCE, and compare.
        RECENT_ARG="--input-recent-Ks 1,2,5,10,20 --include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns --pattern-n-trees 1"
        TAG="bank_multi_flanking_legaltrees_patterntrees"
        echo "NOTE: combined bank not fully implemented — this variant"
        echo "runs pattern_trees; use bank_multi_flanking_legaltrees separately"
        echo "for the legal-tree comparison numbers."
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
# For any pattern-tree variant (--tree-target patterns), skip BCE/probOR on
# the enormous 47K-unit hidden layer (~2h each × 5 seeds), and only train
# Linear->ProbOR (the readout we care about).  State-tree variants keep the
# full 8-mode default so DerivS + StatPO + BCE etc. all get their numbers.
if [[ "${RECENT_ARG:-}" == *"--tree-target patterns"* ]] \
        && [ -z "${LEGAL_MODES_OVERRIDE:-}" ]; then
    LEGAL_MODES_OVERRIDE="patterns_structured_probor"
fi
# Pass PICKLE_DIR env var to load synthetic games from disk instead of
# playing them from scratch.  Enables scaling to millions of games in
# minutes instead of days.
PICKLE_ARG=""
if [ -n "${PICKLE_DIR:-}" ]; then
    PICKLE_ARG="--pickle-dir ${PICKLE_DIR}"
fi
# Pattern-tree fitting is embarrassingly parallel over 960 (or 9600 with
# bagging) trees.  State/legal per-cell trees are fewer (64) but each
# joblib worker holds a copy of Xnp, so we default to 1 to avoid OOM.
TREE_N_JOBS=${TREE_N_JOBS:-1}
case "${TAG}" in
    pattern_trees*)
        TREE_N_JOBS=${TREE_N_JOBS_PATTERN:-8}
        ;;
esac

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
    --tree-n-jobs ${TREE_N_JOBS} \
    --top-k-per-cell ${TOP_K} \
    --hidden-activation ${HIDDEN_ACTIVATION:-relu} \
    --probe-epochs 100 \
    --probe-seeds 5 \
    --task both \
    --tree-target ${TREE_TARGET} \
    --legal-modes ${LEGAL_MODES_OVERRIDE:-bce,probor,derived,state_probor,patterns_probor,patterns_structured_probor,cells_structured_probor,patterns_linear_probor} \
    --legal-probe-epochs 100 \
    ${RECENT_ARG} \
    ${PICKLE_ARG} \
    --device cuda \
    --cache-tr ${CACHE_TR} \
    --cache-te ${CACHE_TE} \
    ${EXTRA_ARGS:-} \
    --out ${OUT}

echo "Completed at: $(date)"
