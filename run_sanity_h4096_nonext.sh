#!/bin/bash
# Sanity check: run plot_mlp_overall_by_turn.py on the ORIGINAL (non-ext)
# H=4096 movegrid probe. Table says 93.41% over turns 5-53. If we get that,
# eval script is correct -> probe_..._ext_3ep.pt is broken. If we don't,
# the eval script itself is buggy on move_grid features.
#
# Usage: sbatch run_sanity_h4096_nonext.sh

#SBATCH --job-name=sanity_h4k
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sanity_h4k_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs experiments/plots
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python plot_mlp_overall_by_turn.py \
    --pat-ckpt "$CKPT_DIR/pattern_simple_direct_H4096_move_grid.pt" \
    --probe-ckpt "$CKPT_DIR/probe_direct_H4096_move_grid.pt" \
    --hidden 4096 --features move_grid \
    --output experiments/plots/sanity_h4096_movegrid_by_turn.png
