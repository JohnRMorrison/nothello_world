#!/bin/bash
# Per-cell linear probe on H=1024 wheneven hidden layer.
#
# Decides between "uniform correlated errors" (tight per-cell distribution -
# architectural change needed) and "specific weak cells" (wide distribution -
# targeted training works) as the cause of the long-pattern recall gap.
#
# Usage: sbatch run_probe_per_cell.sh

#SBATCH --job-name=probe_cell
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_cell_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H1024_wheneven.pt

echo "============================================"
echo "Per-cell linear probe on H=1024 wheneven hidden"
echo "Checkpoint: $CKPT"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_per_cell.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 1024 \
    --n-train 40000 --n-test 20000 \
    --output logs/probe_per_cell_H1024_wheneven.npz

echo "Completed: $(date)"
