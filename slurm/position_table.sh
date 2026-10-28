#!/bin/bash
#SBATCH --job-name=position_table
#SBATCH --output=logs/position_table_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=06:00:00
#
# Per-position table of intervention vs natural-play probabilities,
# Section 5.4 positions, Nanda et al.'s push, fixed size 1x and 2.5x.

FIXED_1X=1.530237
declare -A EXPECTED=(
    [build_position_table.py]=d8d3bea76858b6bd510d9921ecbb1c11
    [generate_paper_numbers.py]=d9dc6553bf981a3b488741ea0aed0511
    [src/intervention.py]=d3c321c0c14026ab7b6b08adbbde7f70
)

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
mkdir -p logs "Intervention Results/position_table"

echo "node $(hostname)"
for f in "${!EXPECTED[@]}"; do
    got=$(md5sum "$f" | awk '{print $1}')
    echo "$f $got"
    if [[ "$got" != "${EXPECTED[$f]}" ]]; then
        echo "!!! STALE CODE on $(hostname): $f does not match. Resubmit excluding this node."
        exit 1
    fi
done

python -u build_position_table.py --fixed-alpha $FIXED_1X --alpha-mults 1 2.5 \
    --out "Intervention Results/position_table/position_table" 2>&1
