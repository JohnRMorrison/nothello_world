#!/bin/bash
# Train the policy head: board state logits → next move
# Usage: sbatch train_policy.sh

#SBATCH --job-name=policy
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_policy_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Training policy head on real Othello games"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python generate_adversarial_games.py \
    --train-policy \
    --max-games 100000 \
    --policy-epochs 20

echo "Completed at: $(date)"
