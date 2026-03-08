#!/bin/bash
# ============================================================================
# Train MLP on move-history features (180-d) with large game datasets
# ============================================================================
#
# Usage:
#   sbatch mlp_feature_search.sh
#   sbatch mlp_feature_search.sh files=50 hidden=1024,2048
#
# Memory: ~35GB per 1M games (180-d features × 49 positions × float32)
#   20 files (~2M games) → ~70GB features + labels + model ≈ 100GB
#   30 files (~3M games) → ~105GB features + labels + model ≈ 140GB
# ============================================================================

#SBATCH --job-name=mlp_feat
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_feat_%j.out
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

# Parse arguments (defaults: 10 files ~1M games, H=1024 and H=2048)
MAX_FILES=10
HIDDEN_DIMS="1024 2048"
for arg in "$@"; do
    if [[ "$arg" == files=* ]]; then
        MAX_FILES="${arg#files=}"
    elif [[ "$arg" == hidden=* ]]; then
        HIDDEN_DIMS="${arg#hidden=}"
        HIDDEN_DIMS="${HIDDEN_DIMS//,/ }"
    fi
done

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Max files: ${MAX_FILES}"
echo "Hidden dims: ${HIDDEN_DIMS}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(python --version 2>&1)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
    --experiment mlp \
    --max-files $MAX_FILES \
    --max-games 99999999 \
    --mlp-hidden $HIDDEN_DIMS \
    --mlp-only \
    --output-dir experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "Completed at: $(date)"
