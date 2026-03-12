#!/bin/bash
# Precompute board states from MLP (run once before sweep jobs)
# Usage: sbatch precompute_board_states.sh

#SBATCH --job-name=precompute
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/precompute_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Precompute board states for 6M games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python generate_adversarial_games.py \
    --precompute-logits \
    --max-games 6000000

echo "Completed at: $(date)"
