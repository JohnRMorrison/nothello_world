#!/bin/bash
#SBATCH --job-name=div
#SBATCH --output=logs/div_%A_%a.out
#SBATCH --error=logs/div_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax00

# Tasks 0-9:   Direction 1 (on experimental) — corruption alphas
# Tasks 10-16: Direction 1 (on experimental) — variants
# Tasks 17-23: Direction 2 (on standard)     — variants

CORRUPTION_LABELS=(alpha000 alpha001 alpha002 alpha005 alpha010 alpha020 alpha030 alpha050 alpha070 alpha100)
VARIANTS=(no_same_quadrant no_diagonal_flips no_row_flips locked_flips max_three_flips self_flanking delayed_flips)

TASK=$SLURM_ARRAY_TASK_ID

echo "Task: ${TASK}"
echo "Started at: $(date)"

source activate othello

if [ $TASK -lt 10 ]; then
    # Direction 1: corruption on experimental games
    LABEL=${CORRUPTION_LABELS[$TASK]}
    GAMES_DIR="experiments/corruption_v2/games_2m/${LABEL}"
    echo "Direction 1 — Corruption: ${LABEL}"
    python measure_legal_divergence.py \
        --games-dir ${GAMES_DIR} \
        --condition-type corruption \
        --condition-label ${LABEL} \
        --direction on_experimental \
        --max-games 100000 \
        --output-dir experiments/divergence

elif [ $TASK -lt 17 ]; then
    # Direction 1: variant on experimental games
    VIDX=$((TASK - 10))
    VARIANT=${VARIANTS[$VIDX]}
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
    VIDX=$((TASK - 17))
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
