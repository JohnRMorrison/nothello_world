#!/bin/bash
#SBATCH --job-name=nanda_push_sec54
#SBATCH --output=logs/nanda_push_sec54_%A_%a.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#
# Section 5.4 numbers with Nanda et al.'s push (add a fixed multiple of the
# target-state probe vector), intervening at layers 4 onward (blocks 3..7).
#   task 0: size 1x    task 1: size 2.5x
# Each task runs our calibrated version first, then their fixed version,
# whose 1x is the median calibrated size printed by the first run.

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
OUT="Intervention Results/nanda_push"
mkdir -p logs "$OUT"

MULTS=(1 2.5); MULT=${MULTS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: size ${MULT}x, node $(hostname)"
md5sum generate_paper_numbers.py src/intervention.py

CAL_OUT="$OUT/paper_numbers_add_calibrated_${MULT}x.txt"
python -u generate_paper_numbers.py --n-positions 5000 \
    --cascade-start-li 4 --later-layers always --push add \
    --size-mode calibrated --alpha-mult $MULT \
    --out "$CAL_OUT" 2>&1 || { echo "!!! calibrated run FAILED"; exit 1; }

FIXED=$(grep "FIXED_1X_SIZE" "$CAL_OUT" | awk '{print $NF}')
if [[ -z "$FIXED" ]]; then echo "!!! could not read FIXED_1X_SIZE"; exit 1; fi
echo "fixed 1x size = $FIXED"

python -u generate_paper_numbers.py --n-positions 5000 \
    --cascade-start-li 4 --later-layers always --push add \
    --size-mode fixed --fixed-alpha $FIXED --alpha-mult $MULT \
    --out "$OUT/paper_numbers_add_fixed_${MULT}x.txt" 2>&1 \
    || { echo "!!! fixed run FAILED"; exit 1; }
