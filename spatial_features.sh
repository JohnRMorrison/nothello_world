#!/bin/bash
# Train MLP on 300-d features (180 base + 120 spatial)
# Usage: sbatch spatial_features.sh

#SBATCH --job-name=spatial
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/spatial_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Spatial Features: 300-d, H=1024 and H=2048"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python spatial_features.py \
    --hidden 1024 2048 \
    --epochs 10 \
    --max-games 1000000 \
    --precomputed

echo "Completed at: $(date)"
