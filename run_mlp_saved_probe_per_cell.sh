#!/bin/bash
#SBATCH --job-name=mlp_probe_pc
#SBATCH -c 2
#SBATCH --time=00:15:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_probe_pc_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_wheneven.pt
PROBE=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/probe_direct_H512_wheneven.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python analyze_mlp_saved_probe_per_cell.py \
    --ckpt "$CKPT" --probe "$PROBE" --hidden 512
