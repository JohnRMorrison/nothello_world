#!/bin/bash
# Train a Nanda-style parity-split board-state probe on the next-rule MLP.
#
# Prereq: run_train_next_rule.sh must have produced
#         next_rule_H{H}_{features}.pt first.
#
# Usage:
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> run_probe_next_rule.sh

#SBATCH --job-name=probe_nr
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_nr_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/next_rule_H512_wheneven.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_next_rule.py \
    --ckpt "$CKPT" \
    --features when+even \
    --hidden 512 --epochs 5
