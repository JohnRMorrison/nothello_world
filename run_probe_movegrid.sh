#!/bin/bash
# Train Nanda-style probe on the move_grid H=512 pattern detector's hidden.
# Once trained, lets us check if the rich move-history input (3600-d full
# move sequence) gives the MLP enough info to maintain board state across
# turns -- vs the 120-d when+even input which can't track flips.
#
# Usage: sbatch run_probe_movegrid.sh

#SBATCH --job-name=probe_mg
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_mg_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_move_grid.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 512 --epochs 5
