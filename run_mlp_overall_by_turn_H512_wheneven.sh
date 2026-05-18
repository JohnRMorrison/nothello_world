#!/bin/bash
# Per-turn overall (all-64-cells) decoding accuracy for the H=512 wheneven
# pattern detector + matching probe. Fills the per-turn analog of the 91.25%
# headline number from the wheneven table.
#
# Usage: sbatch run_mlp_overall_by_turn_H512_wheneven.sh

#SBATCH --job-name=mlp_byturn_h512we
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_byturn_h512we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs experiments/plots
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python plot_mlp_overall_by_turn.py \
    --pat-ckpt   "$CKPT_DIR/pattern_detector_checkpoints/pattern_simple_direct_H512_wheneven.pt" \
    --probe-ckpt "$CKPT_DIR/pattern_detector_checkpoints/probe_direct_H512_wheneven.pt" \
    --hidden 512 --features when+even
