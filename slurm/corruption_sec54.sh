#!/bin/bash
#SBATCH --job-name=corruption_sec54
#SBATCH --output=logs/corruption_sec54_%A_%a.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#
# Corruption measures on the Section 5.4 positions, Nanda et al.'s push with
# their fixed size, layers 4 onward.  task 0: 1x   task 1: 2.5x
# Fixed 1x size = median calibrated size from nanda_push/paper_numbers_add_calibrated_1x.txt

FIXED_1X=1.530237
EXPECTED_GPN=d9dc6553bf981a3b488741ea0aed0511
EXPECTED_INTV=d3c321c0c14026ab7b6b08adbbde7f70

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
OUT="Intervention Results/corruption"
mkdir -p logs "$OUT"

MULTS=(1 2.5); MULT=${MULTS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: size ${MULT}x, node $(hostname)"
GOT_GPN=$(md5sum generate_paper_numbers_corr.py | awk '{print $1}')
GOT_INTV=$(md5sum src/intervention.py | awk '{print $1}')
echo "generate_paper_numbers_corr.py $GOT_GPN   src/intervention.py $GOT_INTV"
if [[ "$GOT_GPN" != "$EXPECTED_GPN" || "$GOT_INTV" != "$EXPECTED_INTV" ]]; then
    echo "!!! STALE CODE on $(hostname): checksums do not match. Resubmit excluding this node."
    exit 1
fi

python -u generate_paper_numbers_corr.py --n-positions 5000 \
    --cascade-start-li 4 --later-layers always --push add \
    --size-mode fixed --fixed-alpha $FIXED_1X --alpha-mult $MULT \
    --corruption \
    --save-records "$OUT/records_add_fixed_${MULT}x.npz" \
    --out "$OUT/corruption_add_fixed_${MULT}x.txt" 2>&1
