#!/bin/bash
# Overnight chain: WAIT for the currently-running 1-epoch grid to finish, then
# launch the cached converged (multi-epoch) re-run -- no manual step at 3am.
#
# Completion = no `train_streaming_probe.py` processes remain (covers both
# clean finishes and crashes, so it can never wait forever).  The 1-epoch
# outputs are archived first so the converged run doesn't overwrite them.
#
#   nohup bash run_overnight_chain.sh [CHUNK_DIR] [GAMES] [EPOCHS] [CONFIGS] \
#       > logs/overnight_chain.out 2>&1 &   disown
#   # ...or just run it in a spare tmux window (tmux persists on the pod).
set -u
CH=${1:-/workspace/feature_chunks}
GAMES=${2:-3000000}
EPOCHS=${3:-4}
CONFIGS=${4:-"J1_A J1_B J2_A J2_B J3_A J3_B"}
ALL="J0_A J1_A J1_B J2_A J2_B J3_A J3_B"
mkdir -p stream_out_1ep logs/1ep

ts () { date '+%Y-%m-%d %H:%M:%S'; }
echo "[chain $(ts)] waiting for the current grid (train_streaming_probe.py) to exit..."

# Poll until two consecutive zero-process readings (avoids acting during a
# momentary gap while configs hand off).
zero=0
while [ "$zero" -lt 2 ]; do
  sleep 120
  if pgrep -f train_streaming_probe.py > /dev/null 2>&1; then
    zero=0
  else
    zero=$((zero + 1))
  fi
done
echo "[chain $(ts)] current grid fully exited."

# Archive the 1-epoch results so the converged run starts clean.
for f in ${ALL}; do
  [ -f "stream_out/${f}.pt" ]      && mv "stream_out/${f}.pt"      "stream_out_1ep/${f}.pt"
  [ -f "logs/stream_${f}.out" ]    && cp "logs/stream_${f}.out"    "logs/1ep/stream_${f}.out"
done
echo "[chain $(ts)] 1-epoch results archived -> stream_out_1ep/ , logs/1ep/"
echo "[chain $(ts)] 1-epoch legal metrics (for reference):"
for f in ${ALL}; do
  line=$(grep -E "LEGAL-MOVE" "logs/1ep/stream_${f}.out" 2>/dev/null | tail -1)
  printf "    %-8s %s\n" "$f" "${line:-<no result>}"
done

echo "[chain $(ts)] launching cached converged grid: epochs=${EPOCHS}  configs=${CONFIGS}"
bash run_stream_grid_cached.sh "${CH}" "${GAMES}" "${EPOCHS}" "${CONFIGS}"
echo "[chain $(ts)] DONE."
