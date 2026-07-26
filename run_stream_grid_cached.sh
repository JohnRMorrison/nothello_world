#!/bin/bash
# CONVERGED re-run with a SHARED on-disk chunk cache.
#
# Build each chunk's method-agnostic 181-d cache ONCE (serially), then train
# the requested configs for EPOCHS epochs reading the cache — so multi-epoch /
# multi-config costs ONE load (+ 960-pattern compute) per chunk, not N.  This
# is the cheap way to get the *converged* numbers the 1-epoch grid only floored.
#
#   bash run_stream_grid_cached.sh [CHUNK_DIR] [GAMES] [EPOCHS] [CONFIGS]
#     CONFIGS: space-sep subset of {J0_A J1_A J1_B J2_A J2_B J3_A J3_B}
#              default = J1/J2/J3 x {A,B} (drop J0 flanking bar).
#   env: CACHE=<dir>   MAXPOS=<rows/chunk>
#
# To use ALL the games at full width, raise MAXPOS (e.g. 28000000) and run
# fewer configs at once so RAM holds.
set -u
CH=${1:-/workspace/feature_chunks}
GAMES=${2:-3000000}
EPOCHS=${3:-4}
CONFIGS=${4:-"J1_A J1_B J2_A J2_B J3_A J3_B"}
CACHE=${CACHE:-/workspace/chunk_cache}
MAXPOS=${MAXPOS:-10000000}
mkdir -p stream_out logs "${CACHE}"
echo "CACHED GRID: chunks=${CH} games=${GAMES} epochs=${EPOCHS} maxpos=${MAXPOS}"
echo "  cache=${CACHE}"
echo "  configs: ${CONFIGS}"

BASE="--no-recent --data-source chunk-ext --chunk-dir ${CH} \
  --canonicalize-mover --ply-min 5 --ply-max 54 --probe-type linpo \
  --num-train-games ${GAMES} --num-test-games 100000 --epochs ${EPOCHS} \
  --max-positions-per-file ${MAXPOS} --cache-dir ${CACHE}"

# --- Step 1: build the shared cache ONCE (serial), then all configs reuse it.
echo "=== precomputing shared chunk cache (serial) ==="
python -u train_streaming_probe.py ${BASE} --flanking-only \
  --precompute-cache-only --out /dev/null 2>&1 | tee logs/precompute_cache.out
echo "=== cache built; launching: ${CONFIGS} ==="

run () {   # $1=label (Jx_A|Jx_B)
  local label=$1 load extra=""
  case "$label" in
    J0_*) load="--flanking-only" ;;
    J1_*) load="--load-trees-from banks/J1_perpattern.pt --no-flanking" ;;
    J2_*) load="--load-trees-from banks/J2_grouped.pt --no-flanking" ;;
    J3_*) load="--load-trees-from banks/J3_ordinal.pt --no-flanking" ;;
    *)    echo "  ?? unknown config ${label}"; return ;;
  esac
  case "$label" in *_B) extra="--pattern-bce" ;; esac
  python -u train_streaming_probe.py ${BASE} ${load} ${extra} \
    --out stream_out/${label}.pt > logs/stream_${label}.out 2>&1 &
  echo "  launched ${label} (pid $!)"
}

for c in ${CONFIGS}; do run "$c"; done
echo "waiting..."; wait
echo "=================================================="
echo "############  CACHED GRID RESULTS (prob-OR legal, final epoch)  ############"
printf "%-8s %10s %10s\n" "config" "legal_rec" "legal_F1"
for f in ${CONFIGS}; do
  log="logs/stream_${f}.out"
  rec=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | tail -1 | grep -oE "recall=[0-9.]+%" | grep -oE "[0-9.]+%")
  f1=$(grep -E "LEGAL-MOVE" "$log" 2>/dev/null | tail -1 | grep -oE "F1=[0-9.]+%" | grep -oE "[0-9.]+%")
  printf "%-8s %10s %10s\n" "$f" "${rec:-?}" "${f1:-?}"
done
