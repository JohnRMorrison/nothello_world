#!/bin/bash
#SBATCH --job-name=dist_sec54
#SBATCH --output=logs/dist_sec54_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -c 4
#SBATCH --time=06:00:00
#
# Same run as corruption_sec54.sh at 2.5x (Nanda et al.'s push, fixed size,
# layers 4 onward), plus the full 64-square move distribution after every
# intervention (P_after_64) and with no intervention (P_clean_64, per position).
# Written to a separate folder so the existing records file is not overwritten.

FIXED_1X=1.530237
MULT=2.5
EXPECTED_GPN=683351de6a531ea73b524e1999ec0904
EXPECTED_INTV=d3c321c0c14026ab7b6b08adbbde7f70

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
OUT="Intervention Results/corruption/with_dist"
mkdir -p logs "$OUT"

echo "size ${MULT}x, node $(hostname)"
GOT_GPN=$(md5sum generate_paper_numbers_dist.py | awk '{print $1}')
GOT_INTV=$(md5sum src/intervention.py | awk '{print $1}')
echo "generate_paper_numbers_dist.py $GOT_GPN   src/intervention.py $GOT_INTV"
if [[ "$GOT_GPN" != "$EXPECTED_GPN" || "$GOT_INTV" != "$EXPECTED_INTV" ]]; then
    echo "!!! STALE CODE on $(hostname): checksums do not match. Resubmit excluding this node."
    exit 1
fi

python -u generate_paper_numbers_dist.py --n-positions 5000 \
    --cascade-start-li 4 --later-layers always --push add \
    --size-mode fixed --fixed-alpha $FIXED_1X --alpha-mult $MULT \
    --corruption \
    --save-records "$OUT/records_add_fixed_${MULT}x.npz" --save-dist \
    --out "$OUT/corruption_add_fixed_${MULT}x.txt" 2>&1

# Should print nothing except possibly timing lines: same numbers as the earlier run.
diff "Intervention Results/corruption/corruption_add_fixed_${MULT}x.txt" \
     "$OUT/corruption_add_fixed_${MULT}x.txt" && echo "MATCHES earlier 2.5x results"
