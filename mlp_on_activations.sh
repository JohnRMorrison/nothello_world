#!/bin/bash
# ============================================================================
# Train MLPs on Othello-GPT activations (varying hidden dim)
# ============================================================================
#
# Step 1: Extract activations for 1M games (single GPU job)
#   sbatch mlp_on_activations.sh extract
#
# Step 2: Train MLPs on each layer (array job, one per layer)
#   sbatch --array=0-8 mlp_on_activations.sh train
#
# Step 3: Plot results (no GPU needed)
#   python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
#       plot --results-dir cached_acts_1m
# ============================================================================

#SBATCH --job-name=mlp_acts
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_acts_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

ACTS_DIR=activations/activations_synthetic_1m
MODE="${1:-extract}"

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Mode: ${MODE}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

if [[ "$MODE" == "extract" ]]; then
    python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
        extract --max-games 1000000 --output-dir $ACTS_DIR

elif [[ "$MODE" == "train" ]]; then
    LAYER=${SLURM_ARRAY_TASK_ID:-0}
    echo "Training MLPs on layer ${LAYER}"

    CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.mlp_on_activations \
        train --acts-dir $ACTS_DIR --layer $LAYER \
        --hidden-dims 4 8 16 32 64 128 256 512 1024 2048 \
        --epochs 10
fi

echo "Completed at: $(date)"
