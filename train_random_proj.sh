#!/bin/bash
# Train random projection models: frozen random first layer, train output only.
#
# Usage:
#   sbatch --array=0-19 train_random_proj.sh
#
# Tasks 0-9:  H=512,  seeds 0-9
# Tasks 10-19: H=1024, seeds 0-9

#SBATCH --job-name=randproj
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/randproj_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TASK=${SLURM_ARRAY_TASK_ID:-0}

if [ $TASK -lt 10 ]; then
    HIDDEN=512
    SEED=$TASK
else
    HIDDEN=1024
    SEED=$((TASK - 10))
fi

echo "============================================"
echo "Random projection: H=$HIDDEN seed=$SEED"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train_streaming.py \
    --features when \
    --hidden $HIDDEN \
    --epochs 1 \
    --random-proj \
    --seed $SEED

echo "Completed at: $(date)"
