#!/bin/bash
# ============================================================================
# Probe Othello-GPT via TransformerLens — one job per layer
# ============================================================================

#SBATCH --job-name=probe_tl
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/probe_tl_%A_layer%a.out
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

LAYER=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Probing layer: ${LAYER} (resid_post via TransformerLens)"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

# First ensure transformer_lens is installed
pip install transformer_lens 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.probe_tl_othello \
    --layer $LAYER \
    --max-games 100000 \
    --probe-epochs 2 \
    --output-dir experiments/mathematical_transformation_experiments/tl_probe_results

echo "Completed at: $(date)"
