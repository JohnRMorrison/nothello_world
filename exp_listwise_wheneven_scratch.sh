#!/bin/bash
# Experiment 2: H=512 + 120-d (when+even), pattern BCE + listwise from scratch.
# Trains the 960-pattern model from scratch with a multi-task loss:
#   pattern BCE  +  1.0 * listwise CE on logsumexp-aggregated cells.
# The listwise term is what specifically targets recall@K.
#
# Usage: sbatch exp_listwise_wheneven_scratch.sh

#SBATCH --job-name=pat_listw_we
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_listw_we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Experiment 2: H=512 + when+even, pattern BCE + listwise (lw=1.0)"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 512 \
    --features when+even \
    --listwise-weight 1.0 \
    --epochs 3

echo "Completed: $(date)"
