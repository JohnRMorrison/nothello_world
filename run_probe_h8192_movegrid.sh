#!/bin/bash
# Train Nanda-style board-state probe on H=8192 movegrid pattern detector.
# Sequential dependency: train job must finish first.

#SBATCH --job-name=probe_h8k
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_h8k_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H8192_move_grid.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT" --mode direct --hidden 8192 --epochs 5
