#!/bin/bash
# Evaluate next-cell MLP on legal-move prediction.
#
# Usage:
#   sbatch eval_next_cell_legality.sh
#   CKPT=path/to/ckpt.pt CHUNKS=2 sbatch eval_next_cell_legality.sh

#SBATCH --job-name=eval_legal
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_legal_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=${CKPT:-experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/next_cell_mlp_H512_move_grid.pt}
CHUNKS=${CHUNKS:-2}

echo "============================================"
echo "Evaluating next-cell MLP for legality"
echo "CKPT=$CKPT  CHUNKS=$CHUNKS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python eval_next_cell_legality.py \
    --ckpt $CKPT --chunks $CHUNKS

echo "Completed at: $(date)"
