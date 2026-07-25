#!/bin/bash
# Portable tree-only bake-off runner — NO SLURM, NO external data (self-play).
# Runs the 4 Round-1 configs directly with python and prints the comparison.
# Clone the repo on any box (VM / Colab / laptop) and run:
#     bash run_bakeoff.sh [DEVICE] [GAMES]
#   DEVICE = cpu (default) | cuda
#   GAMES  = train games (default 20000; trim to ~8000 on a low-RAM box —
#            the per-pattern/ordinal configs build 48k-wide activations)
#
# Needs: python with torch, scikit-learn, numpy.  hand_crafted_flanking_patterns.pt
# is tracked in the repo, so nothing else to download.
set -u
DEVICE=${1:-cpu}
GAMES=${2:-20000}
TEST=$(( GAMES / 4 ))
OUTDIR=bakeoff_out
mkdir -p "$OUTDIR" logs
COMMON="--include-flanking-patterns hand_crafted_flanking_patterns.pt \
  --tree-target patterns --canonicalize-mover --skip-state-probe \
  --num-train-games ${GAMES} --num-test-games ${TEST} --ply-min 5 --ply-max 54 \
  --tree-max-depth 15 --tree-min-samples-leaf 50 \
  --task legal --legal-modes bce --legal-probe-epochs 100 --probe-seeds 1 \
  --device ${DEVICE}"
TREE_ONLY="--no-flanking-features --max-leaf-nodes 50"

run () {   # $1=label  $2...=extra args
  local label=$1; shift
  echo "================ ${label} ================"
  python -u midgame_tree_mlp.py ${COMMON} "$@" \
    --out "${OUTDIR}/${label}.pt" 2>&1 | tee "logs/bakeoff_${label}.out"
}

# J0 flanking-only reference (960 rules, no trees; flanking ARE the features)
run J0_flanking --skip-tree-fit
# J1 per-pattern (no recency)
run J1_perpattern --pattern-n-trees 1 ${TREE_ONLY}
# J2 grouped by target square
run J2_grouped --pattern-tree-mode grouped ${TREE_ONLY}
# J3 time-ordinal split (leaf-based hidden layer)
run J3_ordinal --time-ordinal movesago --time-ordinal-split-color \
  --hidden-from-leaves --pattern-n-trees 1 ${TREE_ONLY}

echo ""
echo "################  RESULTS  ################"
printf "%-16s %14s %14s\n" "config" "hidden_units" "per_cell_acc"
for f in J0_flanking J1_perpattern J2_grouped J3_ordinal; do
  log="logs/bakeoff_${f}.out"
  units=$(grep -oE "total hidden units: *[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  [ -z "$units" ] && units=$(grep -oE "H_tr \([0-9]+, [0-9]+\)" "$log" 2>/dev/null | grep -oE ", [0-9]+" | tr -d ', ' | head -1)
  acc=$(grep -E "BCE .*per-cell acc" "$log" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+%" | head -1)
  printf "%-16s %14s %14s\n" "$f" "${units:-?}" "${acc:-?}"
done
