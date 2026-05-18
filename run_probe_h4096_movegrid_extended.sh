#!/bin/bash
# Train Nanda-style probe on the extended-range H=4096 movegrid pattern
# detector. Uses chunk_ext_*.npz (turns 5-58).

#SBATCH --job-name=probe_h4kx
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_h4kx_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKPT_DIR/pattern_simple_direct_H4096_move_grid_ext.pt" \
    --mode direct --hidden 4096 --epochs 5 \
    --chunk-prefix chunk_ext_

# Rename probe to mark extended-range training
mv "$CKPT_DIR/probe_direct_H4096_move_grid_ext.pt" \
   "$CKPT_DIR/probe_direct_H4096_move_grid_ext_5_58.pt" \
   2>/dev/null || true
