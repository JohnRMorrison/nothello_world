#!/bin/bash
# STAGE 1 of the bake-off: FIT the trees once and SAVE reusable banks.
# No readout here — train readouts later (any number of times) with run_readout.sh.
#   bash run_fit.sh [DEVICE] [GAMES] [JOBS]
# Banks land in banks/<config>.pt (include sklearn trees, so ordinal reloads).
# J0 (flanking) has no trees to fit — it's handled entirely in run_readout.sh.
set -u
DEVICE=${1:-cpu}
GAMES=${2:-20000}
JOBS=${3:-$(nproc)}
TEST=$(( GAMES / 4 ))
mkdir -p banks logs
echo "FIT stage: ${GAMES} games, ${JOBS}-way tree parallelism, device=${DEVICE}"

# feature/tree flags shared by every tree config
BASE="--include-flanking-patterns hand_crafted_flanking_patterns.pt \
  --tree-target patterns --canonicalize-mover --no-flanking-features \
  --num-train-games ${GAMES} --num-test-games ${TEST} --ply-min 5 --ply-max 54 \
  --tree-max-depth 15 --tree-min-samples-leaf 50 --max-leaf-nodes 50 \
  --tree-n-jobs ${JOBS} --skip-state-probe --tree-fit-only --device ${DEVICE}"

fit () {   # $1=label  $2...=extra tree-defining args
  local label=$1; shift
  echo "================ FIT ${label} ================"
  python -u midgame_tree_mlp.py ${BASE} "$@" \
    --out "banks/${label}.pt" 2>&1 | tee "logs/fit_${label}.out"
}

fit J1_perpattern --pattern-n-trees 1
fit J2_grouped    --pattern-tree-mode grouped
fit J3_ordinal    --time-ordinal movesago --time-ordinal-split-color \
  --hidden-from-leaves --pattern-n-trees 1

echo ""
echo "banks saved:"; ls -lh banks/*.pt
