#!/bin/bash
#SBATCH --job-name=divergence
#SBATCH --output=logs/divergence_%A_%a.out
#SBATCH --error=logs/divergence_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-3

VARIANTS=(skip_empty_flips capture_any wrap_flips adjacent_legal)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Started at: $(date)"
echo "Variant: $VARIANT"

source activate othello

python measure_legal_divergence.py \
    --games-dir experiments/variants/games_2m/$VARIANT \
    --condition-type variant \
    --variant-name $VARIANT \
    --output-dir experiments/divergence/

echo "Finished at: $(date)"
