#!/bin/bash
# CONVERGED J3 via the leaf-index fast path (bypasses the GIL-bound tree.apply).
#
#   step 1: build_leaf_cache.py -- apply the 960 ordinal trees ONCE per chunk,
#           process-parallel (fork-shared trees, os.pwrite), saving (N,960)
#           int16 leaf-ids.  ~minutes/chunk instead of ~6.4 hr/chunk.
#   step 2: train J3_A + J3_B for EPOCHS reading the leaf cache -- H is a cheap
#           gather+compare, so training runs at ~J1 speed (no tree.apply).
#
#   bash run_j3_converged.sh [CHUNK_DIR] [GAMES] [EPOCHS] [PROCS]
#   env: CACHE=<181-cache dir>  LEAF=<leaf-cache dir>  MAXPOS=<rows/chunk>
#
# Free the box first (kill the old configs) so the parallel apply gets the cores.
set -u
CH=${1:-/workspace/feature_chunks}
GAMES=${2:-3000000}
EPOCHS=${3:-4}
PROCS=${4:-96}
CACHE=${CACHE:-/workspace/chunk_cache}
LEAF=${LEAF:-/workspace/leaf_cache}
MAXPOS=${MAXPOS:-10000000}
mkdir -p stream_out logs "${LEAF}"

echo "=== step 1: build leaf-index cache (process-parallel apply, procs=${PROCS}) ==="
python -u build_leaf_cache.py --bank banks/J3_ordinal.pt \
  --chunk-dir "${CH}" --cache-dir "${CACHE}" --leaf-cache-dir "${LEAF}" \
  --load-cap "${MAXPOS}" --eval-cap 500000 --procs "${PROCS}" \
  2>&1 | tee logs/build_leaf_cache.out
if [ ! -f "${LEAF}/colmap.npz" ]; then
  echo "!! leaf cache build failed (no colmap.npz) -- aborting"; exit 1
fi

BASE="--no-recent --data-source chunk-ext --chunk-dir ${CH} \
  --canonicalize-mover --ply-min 5 --ply-max 54 --probe-type linpo \
  --num-train-games ${GAMES} --num-test-games 100000 --epochs ${EPOCHS} \
  --max-positions-per-file ${MAXPOS} --load-trees-from banks/J3_ordinal.pt \
  --no-flanking --leaf-index-cache-dir ${LEAF}"

echo "=== step 2: train J3_A + J3_B off the leaf cache (gather+compare H) ==="
python -u train_streaming_probe.py ${BASE} \
  --out stream_out/J3_A_conv.pt > logs/stream_J3_A_conv.out 2>&1 &
echo "  launched J3_A_conv (pid $!)"
python -u train_streaming_probe.py ${BASE} --pattern-bce \
  --out stream_out/J3_B_conv.pt > logs/stream_J3_B_conv.out 2>&1 &
echo "  launched J3_B_conv (pid $!)"
wait

echo "=================================================="
echo "############  CONVERGED J3 (prob-OR legal, per epoch)  ############"
for f in J3_A_conv J3_B_conv; do
  echo "--- $f ---"
  grep -h "LEGAL-MOVE" "logs/stream_${f}.out" 2>/dev/null || echo "(no result)"
done
