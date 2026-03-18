#!/bin/bash
#SBATCH --job-name=imp_gen
#SBATCH --output=logs/imp_gen_%A_%a.out
#SBATCH --error=logs/imp_gen_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-5

# GER targets: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
GERS=(0.0 0.1 0.2 0.3 0.4 0.5)

GER=${GERS[$SLURM_ARRAY_TASK_ID]}
# Convert to label: 0.1 -> ger010
GER_LABEL=$(printf "ger%03d" $(echo "$GER * 100" | bc | cut -d. -f1))
OUTPUT_DIR="experiments/impossible/games_2m/${GER_LABEL}"

echo "GER target: ${GER} (label: ${GER_LABEL})"
echo "Started at: $(date)"

source activate othello

python generate_impossible_games.py \
    --target-ger ${GER} \
    --num-games 2000000 \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

echo "Finished at: $(date)"
