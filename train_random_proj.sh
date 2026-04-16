#!/bin/bash
# Train random projection models: frozen random first layer, train output only.
#
# Usage:
#   sbatch train_random_proj.sh 2048        # single run, H=2048
#   sbatch train_random_proj.sh 2048 3      # H=2048, 3 epochs
#   sbatch train_random_proj.sh 2048 2 5    # H=2048, 2 epochs, seed=5
#
# Or array mode (uses positional args for H and epochs, array ID for seed):
#   sbatch --array=0-2 train_random_proj.sh 1024 2   # H=1024, 2 epochs, seeds 0-2

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

HIDDEN=${1:-1024}
EPOCHS=${2:-2}
SEED=${3:-${SLURM_ARRAY_TASK_ID:-0}}

echo "============================================"
echo "Random projection: H=$HIDDEN seed=$SEED epochs=$EPOCHS"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train_streaming.py \
    --features when \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --random-proj \
    --seed $SEED

echo "Completed at: $(date)"
