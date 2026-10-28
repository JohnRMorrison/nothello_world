#!/bin/bash
#SBATCH --job-name=cascade_sweep
#SBATCH --output=logs/cascade_sweep_%A_%a.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --array=0-12
#
# Usage:  sbatch slurm/cascade_sweep.sh skip_if_target
#         sbatch slurm/cascade_sweep.sh always
#
# Task 0-7  : sweep 1, Li starting layer 1..8 at 1x
#             (+ Figure 7 benchmark for that starting layer;
#              starting layer 4 gets every size of sweep 2)
# Task 8-12 : sweep 2, Li starting layer 4 at 1.5x 2x 2.5x 3x 4x
#             (1x for layer 4 is task 3)

MODE="$1"
if [[ "$MODE" != "skip_if_target" && "$MODE" != "always" ]]; then
    echo "usage: sbatch slurm/cascade_sweep.sh {skip_if_target|always}"; exit 1
fi

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
mkdir -p logs "Intervention Results/cascade"

T=$SLURM_ARRAY_TASK_ID
SIZES4="1 1.5 2 2.5 3 4"
if (( T <= 7 )); then
    LI=$((T + 1)); MULT=1
else
    LI=4; MULTS=(1.5 2 2.5 3 4); MULT=${MULTS[$((T - 8))]}
fi
echo "task $T: Li start layer $LI, size ${MULT}x, later_layers=$MODE, node $(hostname)"
# Fingerprint of the code this node actually reads (expected values are in
# the commit / local md5s); a mismatch means a stale shared-drive copy.
md5sum generate_paper_numbers.py run_li_benchmark.py src/intervention.py

python -u generate_paper_numbers.py \
    --n-positions 5000 \
    --cascade-start-li $LI \
    --later-layers $MODE \
    --alpha-mult $MULT \
    --out "Intervention Results/cascade/paper_numbers_cascade_Li${LI}_${MULT}x_${MODE}.txt" 2>&1
GPN_STATUS=$?
if (( GPN_STATUS != 0 )); then
    echo "!!! generate_paper_numbers.py FAILED (exit $GPN_STATUS)"
fi

if (( T <= 7 )); then
    if (( LI == 4 )); then BM_MULTS="$SIZES4"; else BM_MULTS="1"; fi
    python -u run_li_benchmark.py \
        --cascade-start-li $LI \
        --later-layers $MODE \
        --alpha-mults $BM_MULTS \
        --out-suffix "_cascade_Li${LI}_${MODE}" 2>&1
fi

# Make a Section 5.4 failure show up as FAILED in sacct, not COMPLETED.
exit $GPN_STATUS
