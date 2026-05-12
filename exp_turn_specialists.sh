#!/bin/bash
# SLURM array: train 10 turn-specialist MLPs on positions at fixed turns.
# Tests whether per-turn specialization closes the recall@K gap (training-
# data dilution hypothesis) or doesn't (architectural ceiling hypothesis).
#
# Turns chosen: 5 10 15 20 25 30 35 40 45 50 -- spans the U-shape of
# recall@K in the unified MLP.
#
# Usage: sbatch --array=0-9 exp_turn_specialists.sh

#SBATCH --job-name=turn_spec
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/turn_spec_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TURNS=(5 10 15 20 25 30 35 40 45 50)
T=${TURNS[$SLURM_ARRAY_TASK_ID]}

echo "Specialist for turn=$T (array task $SLURM_ARRAY_TASK_ID)"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_turn_specialist.py \
    --turn "$T" --hidden 512 --epochs 3 --no-exclude-forfeit

# Note: --no-exclude-forfeit disables inline forfeit checking because
# replaying each position's prefix would add ~17h per training run.
# Forfeit contamination is <1% at turns <=50 (T50 specialist sees ~5%),
# so the effect should be small. If results warrant, follow up with a
# precomputed forfeit mask (precompute_forfeit_mask.py, ~5h once) and
# re-run with forfeit exclusion enabled.
