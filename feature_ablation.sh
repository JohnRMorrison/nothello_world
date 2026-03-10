#!/bin/bash
# Feature ablation: which of played/when/even matter?
# Usage: sbatch --array=0-6 feature_ablation.sh

#SBATCH --job-name=feat_abl
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/feat_ablation_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

# Subsets: 0=when, 1=played+when, 2=when+even, 3=played+even,
#          4=played, 5=even, 6=all
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================"
echo "Feature Ablation - subset $TASK_ID"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python feature_ablation.py \
    --subset-id $TASK_ID \
    --max-games 1000000 \
    --epochs 4 \
    --hidden 1024 \
    --precomputed

echo "Completed at: $(date)"
