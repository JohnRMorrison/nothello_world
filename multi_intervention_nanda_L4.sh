#!/bin/bash
#SBATCH --job-name=nanda_L4
#SBATCH --output=logs/nanda_L4_%j.out
#SBATCH --error=logs/nanda_L4_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# FAITHFUL NANDA: single intervention at the residual stream AFTER layer 4
# (--layer-intervene 5 in this codebase's indexing: compute_prefix_activations
# runs blocks[:5] -> resid_post of block 4). Uses the FIXED layer-6 probe
# (main_linear_probe.pth) as the direction -- NO per-cell / per-layer calibration,
# NO cascade. --scale 2.0 negates the projected coordinate (= Nanda's "scale 1"
# flip: new_coord = c*(1-scale) = -c at scale 2). Matches cascade_L4's
# seed / n-games / n-values so faithful-Nanda vs faithful-Li (cascade_L4) share
# the same games/positions.
echo "Started at: $(date)"

source activate othello

cd mechanistic_interpretability

python multi_intervention.py \
    --probe-path main_linear_probe.pth \
    --n-games 200 \
    --scale 2.0 \
    --layer-intervene 5 \
    --n-values 1,2,3,8 \
    --output-dir ../experiments/multi_intervention_nanda_L4 \
    --save-probs \
    --seed 42

echo "Finished at: $(date)"
