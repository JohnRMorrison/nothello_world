#!/bin/bash
# Sweep policy head hidden dims: 64, 128, 256, 512, 1024, 2048
# Usage: sbatch --array=0-5 train_policy_sweep.sh

#SBATCH --job-name=pol_sweep
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/policy_sweep_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

HIDDEN_ARRAY=(64 128 256 512 1024 2048)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
HIDDEN=${HIDDEN_ARRAY[$TASK_ID]}

echo "============================================"
echo "Policy head sweep: H=$HIDDEN, 6M games"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python generate_adversarial_games.py \
    --train-policy \
    --max-games 6000000 \
    --policy-epochs 20 \
    --policy-hidden $HIDDEN

echo "Completed at: $(date)"
