#!/bin/bash
# Color-specific flanking patterns experiment.
#
# Tests whether 1920 color-specific (absolute-color) patterns trained on a
# SINGLE MLP are easier to learn than 960 mine/yours patterns trained on the
# even/odd-split MLPs we use elsewhere. Same hidden size (512), same input
# features (when+even, 120-d), same number of epochs, same data.
#
# Baseline for comparison:
#   pattern_simple_direct_H512_wheneven.pt  (top-1 prob_or ≈ 98.35%)
#
# Output checkpoint:
#   pattern_simple_direct_H512_wheneven_colorspec.pt
#
# Usage: sbatch train_color_specific_H512.sh

#SBATCH --job-name=colorspec_H512
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/colorspec_H512_we_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 512 \
    --features when+even \
    --epochs 3 \
    --color-specific

echo "Done. Now evaluate top-K legal accuracy with:"
echo "  python compare_aggregators.py \\"
echo "    --ckpt experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_wheneven_colorspec.pt \\"
echo "    --hidden 512"
