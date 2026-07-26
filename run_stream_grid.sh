#!/bin/bash
# Streaming grid AT SCALE on chunk-ext (real games): J0/J1/J2 x {A,B}.
#   A = default (prob-OR trained end-to-end vs legal mask)
#   B = --pattern-bce (960 sigmoids trained vs pattern firings, prob-OR inference)
#   bash run_stream_grid.sh [CHUNK_DIR] [GAMES] [EPOCHS]
# (J3 added once its streaming path is built + tested.)
set -u
CH=${1:-/workspace/feature_chunks}
GAMES=${2:-3000000}
EPOCHS=${3:-1}
mkdir -p stream_out logs
echo "STREAM GRID: chunks=${CH} games=${GAMES} epochs=${EPOCHS}"

# --max-positions-per-file caps rows loaded per chunk so 7 configs fit in RAM
# (28M rows x 7 configs would OOM ~187GB).  10M/chunk x 2 chunks = 20M positions
# per config — a solid readout scale, and ~10GB/config so all 7 run in parallel.
MAXPOS=${MAXPOS:-10000000}
BASE="--no-recent --data-source chunk-ext --chunk-dir ${CH} \
  --canonicalize-mover --ply-min 5 --ply-max 54 --probe-type linpo \
  --num-train-games ${GAMES} --num-test-games 100000 --epochs ${EPOCHS} \
  --max-positions-per-file ${MAXPOS}"

run () {   # $1=label  $2=bank(or FLANK)  $3...=extra
  local label=$1 bank=$2; shift 2
  local load
  if [ "$bank" = "FLANK" ]; then
    load="--flanking-only"                       # J0: flanking IS the hidden layer
  else
    load="--load-trees-from banks/${bank} --no-flanking"   # tree-only
  fi
  python -u train_streaming_probe.py ${BASE} ${load} "$@" \
    --out stream_out/${label}.pt > logs/stream_${label}.out 2>&1 &
  echo "  launched ${label} (pid $!)"
}

run J0_A FLANK
run J1_A J1_perpattern.pt
run J1_B J1_perpattern.pt --pattern-bce
run J2_A J2_grouped.pt
run J2_B J2_grouped.pt --pattern-bce
run J3_A J3_ordinal.pt              # ordinal: leaf-based reload + movesago from chunks
run J3_B J3_ordinal.pt --pattern-bce

echo "waiting..."; wait
echo "=================================================="
echo "############  STREAM GRID RESULTS (prob-OR legal recall / F1)  ############"
printf "%-8s %10s %10s\n" "config" "legal_rec" "legal_F1"
for f in J0_A J1_A J1_B J2_A J2_B J3_A J3_B; do
  log="logs/stream_${f}.out"
  rec=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | tail -1 | grep -oE "recall=[0-9.]+%" | grep -oE "[0-9.]+%")
  f1=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | tail -1 | grep -oE "F1=[0-9.]+%" | grep -oE "[0-9.]+%")
  printf "%-8s %10s %10s\n" "$f" "${rec:-?}" "${f1:-?}"
done
