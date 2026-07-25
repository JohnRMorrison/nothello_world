#!/bin/bash
# STAGE 2 of the bake-off: train the READOUT on the saved banks (fast; no re-fit).
# Reuses banks/ from run_fit.sh.  Re-run this as often as you like (different
# metrics / game counts / readouts) without touching the expensive tree fits.
#   bash run_readout.sh [DEVICE] [GAMES]
# GAMES must match run_fit.sh's GAMES (same deterministic self-play → the
# reloaded trees see the games they were fit on).
set -u
DEVICE=${1:-cpu}
GAMES=${2:-20000}
TEST=$(( GAMES / 4 ))
mkdir -p bakeoff_out logs
echo "READOUT stage: ${GAMES} games, device=${DEVICE}"

# readout flags shared by every config
RD="--num-train-games ${GAMES} --num-test-games ${TEST} --ply-min 5 --ply-max 54 \
  --task legal --legal-modes patterns_linear_probor --legal-probe-epochs 100 --probe-seeds 1 \
  --skip-state-probe --canonicalize-mover --device ${DEVICE}"
FEAT="--include-flanking-patterns hand_crafted_flanking_patterns.pt --tree-target patterns"

run () {   # $1=label  $2...=extra args
  local label=$1; shift
  echo "================ READOUT ${label} ================"
  python -u midgame_tree_mlp.py ${RD} "$@" \
    --out "bakeoff_out/${label}.pt" 2>&1 | tee "logs/readout_${label}.out"
}

# J0 flanking reference: no bank — flanking patterns ARE the hidden layer
run J0_flanking ${FEAT} --skip-tree-fit
# J1/J2: binary banks reload via W (step forward)
run J1_perpattern ${FEAT} --no-flanking-features --load-trees-from banks/J1_perpattern.pt
run J2_grouped    ${FEAT} --no-flanking-features --load-trees-from banks/J2_grouped.pt
# J3 ordinal: leaf-based bank — needs the same feature flags so Xnp matches,
# reload rebuilds H from the saved sklearn trees automatically
run J3_ordinal ${FEAT} --no-flanking-features \
  --time-ordinal movesago --time-ordinal-split-color --hidden-from-leaves \
  --load-trees-from banks/J3_ordinal.pt

echo ""
echo "################  RESULTS (legal-move metrics)  ################"
printf "%-16s %8s %10s %10s %9s\n" "config" "units" "legal_rec" "legal_F1" "legal_prec"
for f in J0_flanking J1_perpattern J2_grouped J3_ordinal; do
  log="logs/readout_${f}.out"
  units=$(grep -oE "total hidden units: *[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  [ -z "$units" ] && units=$(grep -oE "combined H_tr \([0-9]+, [0-9]+\)" "$log" 2>/dev/null | grep -oE ", [0-9]+" | tr -d ', ' | head -1)
  [ -z "$units" ] && units=$(grep -oE "H_tr \([0-9]+, [0-9]+\)" "$log" 2>/dev/null | grep -oE ", [0-9]+" | tr -d ', ' | tail -1)
  pc=$(grep -E "per-cell acc" "$log" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+%" | head -1)
  rec=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | grep -oE "recall=[0-9.]+%" | grep -oE "[0-9.]+%" | head -1)
  f1=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | grep -oE "F1=[0-9.]+%" | grep -oE "[0-9.]+%" | head -1)
  prc=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | grep -oE "precision=[0-9.]+%" | grep -oE "[0-9.]+%" | head -1)
  printf "%-16s %8s %10s %10s %9s\n" "$f" "${units:-?}" "${rec:-?}" "${f1:-?}" "${prc:-?}"
done
