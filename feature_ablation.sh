#!/bin/bash
# Feature ablation: which of played/when/even matter?
# Usage: sbatch feature_ablation.sh

#SBATCH --job-name=feat_abl
#SBATCH -c 8
#SBATCH --time=6:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/feat_ablation_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Feature Ablation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python feature_ablation.py --max-games 1000000 --epochs 4 --hidden 1024

echo "Completed at: $(date)"
