#!/bin/bash
# ============================================================================
# Derive human-readable heuristics from trained H=1024 MLP
# Usage: sbatch mlp_analysis.sh
# ============================================================================

#SBATCH --job-name=mlp_anlz
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_analysis_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

OUTPUT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "============================================"
echo "MLP Analysis: derive heuristics from H=1024"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
    --experiment mlp_analysis \
    --precomputed \
    --mlp-hidden 1024 \
    --max-games 1000000 \
    --output-dir $OUTPUT_DIR

echo "Completed at: $(date)"
