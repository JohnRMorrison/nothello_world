#!/bin/bash
# ============================================================================
# Probe state predictor for real Othello board state
# ============================================================================
#
# Usage:
#   sbatch probe_state_pred.sh
#   sbatch probe_state_pred.sh --probe-all-layers
# ============================================================================

#SBATCH --job-name=probe_sp
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_sp_%j.out
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

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Python: $(python --version 2>&1)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.probe_state_pred_for_othello \
    --ckpt-dir experiments/mathematical_transformation_experiments/ckpts/state_pred_vseed42_8L_512d \
    --probe-all-layers \
    --max-games 100000 \
    --probe-epochs 15 \
    "$@"

echo "Completed at: $(date)"
