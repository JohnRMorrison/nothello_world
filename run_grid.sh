#!/bin/bash
# Full bake-off grid, ALL configs in PARALLEL on SELF-PLAY (no chunks/volume).
# Each config does: fit trees -> both readouts A (linpo, legal-mask) AND
# B (pat-BCE, pattern-supervised) -> prob-OR legal recall/F1.
#   bash run_grid.sh [DEVICE] [GAMES] [JOBS_PER_FIT]
# Grid: J0 flanking + {perpattern,grouped,ordinal} x max_leaf {25,50,100}.
set -u
DEVICE=${1:-cpu}
GAMES=${2:-4000}          # small enough that all configs' H fit in RAM at once
JOBS=${3:-6}              # tree-fit cores PER config (configs run concurrently)
TEST=$(( GAMES / 4 ))
mkdir -p grid_out logs
echo "GRID: games=${GAMES}  device=${DEVICE}  jobs/fit=${JOBS}  (all configs in parallel)"

COMMON="--include-flanking-patterns hand_crafted_flanking_patterns.pt \
  --tree-target patterns --canonicalize-mover --no-flanking-features \
  --num-train-games ${GAMES} --num-test-games ${TEST} --ply-min 5 --ply-max 54 \
  --tree-max-depth 15 --tree-min-samples-leaf 50 --tree-n-jobs ${JOBS} \
  --skip-state-probe --task legal \
  --legal-modes patterns_linear_probor,patterns_bce_probor \
  --legal-probe-epochs 100 --probe-seeds 1 --device ${DEVICE}"

run () {  # $1=label  $2...=extra tree-defining args
  local label=$1; shift
  python -u midgame_tree_mlp.py ${COMMON} "$@" \
    --out grid_out/${label}.pt > logs/grid_${label}.out 2>&1 &
  echo "  launched ${label} (pid $!)"
}

run J0_flanking --skip-tree-fit
for ML in 25 50 100; do
  run perpattern_ml${ML} --pattern-n-trees 1 --max-leaf-nodes ${ML}
  run grouped_ml${ML}    --pattern-tree-mode grouped --max-leaf-nodes ${ML}
  run ordinal_ml${ML}    --time-ordinal movesago --time-ordinal-split-color \
    --hidden-from-leaves --pattern-n-trees 1 --max-leaf-nodes ${ML}
done

echo "waiting for all configs..."
wait
echo "=================================================="

echo ""
echo "############  GRID RESULTS (prob-OR legal recall / F1)  ############"
printf "%-18s | %-19s | %-19s\n" "config" "A linpo(legal-mask)" "B patBCE(patterns)"
printf "%-18s | %8s %8s | %8s %8s\n" "" "rec" "F1" "rec" "F1"
grab () {  # $1=log  $2=tag(LinPO|PatBCE)  $3=field(recall|F1)
  grep -E "${2}.*LEGAL-MOVE" "$1" 2>/dev/null | grep -oE "${3}=[0-9.]+%" | grep -oE "[0-9.]+%" | head -1
}
for f in J0_flanking perpattern_ml25 grouped_ml25 ordinal_ml25 \
         perpattern_ml50 grouped_ml50 ordinal_ml50 \
         perpattern_ml100 grouped_ml100 ordinal_ml100; do
  log="logs/grid_${f}.out"
  ar=$(grab "$log" "LinPO" "recall"); af=$(grab "$log" "LinPO" "F1")
  br=$(grab "$log" "PatBCE" "recall"); bf=$(grab "$log" "PatBCE" "F1")
  printf "%-18s | %8s %8s | %8s %8s\n" "$f" "${ar:-?}" "${af:-?}" "${br:-?}" "${bf:-?}"
done
