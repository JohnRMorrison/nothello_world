#!/bin/bash
# ============================================================================
# Probe Othello-GPT for variant board states — one job per (layer, variant)
# ============================================================================
#
# 9 layers (0-8) × 6 variants = 54 jobs
# Array index = layer * 6 + variant_idx
#
# Variants:
#   0: normal       3: gravity
#   1: no_flip      4: shuffled
#   2: row_flip     5: vtable_flip
# ============================================================================

#SBATCH --job-name=probe_var
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-53
#SBATCH --output=logs/probe_var_%A_%a.out
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

# Decode 2D index
TASK_ID=$SLURM_ARRAY_TASK_ID
LAYER=$((TASK_ID / 6))
VARIANT_IDX=$((TASK_ID % 6))

VARIANTS=(normal no_flip row_flip gravity shuffled vtable_flip)
VARIANT=${VARIANTS[$VARIANT_IDX]}

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${TASK_ID}"
echo "Layer: ${LAYER}, Variant: ${VARIANT}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.probe_variant_boards \
    --ckpt-path ckpts/gpt_synthetic.ckpt \
    --layer $LAYER \
    --max-games 100000 \
    --variants $VARIANT \
    --output-dir experiments/mathematical_transformation_experiments/variant_probe_results_v2

echo "Completed at: $(date)"
