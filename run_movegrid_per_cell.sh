#!/bin/bash
# Per-cell turn-stratified analysis on the move_grid H=512 model.
# Depends on run_probe_movegrid.sh having produced
# probe_direct_H512_move_grid.pt.
#
# Usage: sbatch --dependency=afterok:<probe_jid> run_movegrid_per_cell.sh

#SBATCH --job-name=mg_pc
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mg_pc_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_move_grid.pt
PROBE=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/probe_direct_H512_move_grid.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python analyze_mlp_saved_probe_per_cell.py \
    --ckpt "$CKPT" --probe "$PROBE" --hidden 512 \
    --features move_grid
