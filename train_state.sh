#!/bin/bash
# ============================================================================
# SLURM Job Script for Training State Predictor
# ============================================================================
#
# Usage:
#   sbatch train_state.sh
#   sbatch train_state.sh epochs=30
# ============================================================================

#SBATCH --job-name=state_pred
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/state_pred_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Load CUDA
module load cuda/11.8.0

# Activate conda environment
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create log directory
mkdir -p logs

cd $SLURM_SUBMIT_DIR

# Parse epochs argument (optional)
TRAIN_EPOCHS=20
for arg in "$@"; do
    if [[ "$arg" == epochs=* ]]; then
        TRAIN_EPOCHS="${arg#epochs=}"
    fi
done

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Epochs: ${TRAIN_EPOCHS}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(python --version 2>&1)"
echo "============================================"

# Delete old vtable so the new sparse one gets generated
rm -f vtables/vtable_seed42.npy

# Train state predictor
CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.train_state_predictor \
    --epochs $TRAIN_EPOCHS \
    --max-files 20

echo "Completed at: $(date)"
