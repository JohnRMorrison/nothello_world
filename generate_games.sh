#!/bin/bash
# Generate adversarial games at various perturbation levels
# Usage: sbatch --array=0-4 generate_games.sh
#
# Task 0: alpha=0.0 (baseline, no perturbation)
# Task 1: alpha=0.1
# Task 2: alpha=0.3
# Task 3: alpha=0.5
# Task 4: alpha=1.0

#SBATCH --job-name=gen_adv
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gen_adv_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

ALPHA_ARRAY=(0.0 0.1 0.3 0.5 1.0)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ALPHA=${ALPHA_ARRAY[$TASK_ID]}

echo "============================================"
echo "Generating adversarial games: alpha=$ALPHA, 100K games"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python generate_adversarial_games.py \
    --generate \
    --alpha $ALPHA \
    --n-games 100000

echo "Completed at: $(date)"
