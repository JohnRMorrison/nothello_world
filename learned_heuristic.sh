#!/bin/bash
# ============================================================================
# Approach 3: Learned heuristic combos — run on cluster with many games
# ============================================================================
#
# Usage:
#   sbatch learned_heuristic.sh
#   sbatch learned_heuristic.sh games=100000
#
# Vectorized heuristic feature computation is ~55 games/s, so:
#   50K games  ~  15 min
#   100K games ~  30 min
# For >20K games, features are saved to disk via memmap and loaded in chunks
# during training. 120GB RAM covers chunk loading + labels + model overhead.
# ============================================================================

#SBATCH --job-name=learn_heur
#SBATCH -c 4
#SBATCH --time=6:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/learned_heuristic_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Load CUDA
module load cuda/11.8.0

# Activate conda environment
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs

cd $SLURM_SUBMIT_DIR

# Parse games argument (default 50000)
MAX_GAMES=50000
for arg in "$@"; do
    if [[ "$arg" == games=* ]]; then
        MAX_GAMES="${arg#games=}"
    fi
done

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Max games: ${MAX_GAMES}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(python --version 2>&1)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
    --experiment learned_heuristic \
    --max-games $MAX_GAMES \
    --output-dir experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "Completed at: $(date)"
