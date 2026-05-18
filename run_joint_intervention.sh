#!/bin/bash
# Joint intervention: OGPT vs MLP on the same positions with matched
# cell-flip modifications. Compute 5 mass-based metrics.

#SBATCH --job-name=joint_int
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/joint_int_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python joint_intervention.py \
    --mlp-ckpt       "$CKPT_DIR/pattern_simple_direct_H4096_move_grid.pt" \
    --mlp-probe-ckpt "$CKPT_DIR/probe_direct_H4096_move_grid.pt" \
    --mlp-hidden 4096 \
    --n-games 500 --scale 3.0 --layer 5 \
    --output logs/joint_intervention_s3.npz
