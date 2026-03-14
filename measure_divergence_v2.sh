#!/bin/bash
#SBATCH --job-name=div2
#SBATCH --output=logs/div2_%A_%a.out
#SBATCH --error=logs/div2_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Tasks 0-3: Direction 1 (on experimental) — new variants
# Tasks 4-7: Direction 2 (on standard)     — new variants

VARIANTS=(adjacent_legal skip_empty_flips capture_any wrap_flips)

TASK=$SLURM_ARRAY_TASK_ID

echo "Task: ${TASK}"
echo "Started at: $(date)"

source activate othello

if [ $TASK -lt 4 ]; then
    # Direction 1: variant on experimental games
    VARIANT=${VARIANTS[$TASK]}
    GAMES_DIR="experiments/variants/games_2m/${VARIANT}"
    echo "Direction 1 — Variant: ${VARIANT}"
    python measure_legal_divergence.py \
        --games-dir ${GAMES_DIR} \
        --condition-type variant \
        --condition-label ${VARIANT} \
        --variant-name ${VARIANT} \
        --direction on_experimental \
        --max-games 100000 \
        --output-dir experiments/divergence

else
    # Direction 2: variant on standard Othello games
    VIDX=$((TASK - 4))
    VARIANT=${VARIANTS[$VIDX]}
    echo "Direction 2 — Variant on standard: ${VARIANT}"
    python measure_legal_divergence.py \
        --games-dir experiments/variants/games_2m/${VARIANT} \
        --condition-type variant \
        --condition-label ${VARIANT} \
        --variant-name ${VARIANT} \
        --std-games-dir experiments/corruption_v2/games_2m/alpha000 \
        --direction on_standard \
        --max-games 100000 \
        --output-dir experiments/divergence
fi

echo "Finished at: $(date)"
