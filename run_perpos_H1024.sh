#!/bin/bash
# Run compare_aggregators.py on the H=1024 wheneven checkpoint with
# --save-per-position to dump per-cell scores for error-distribution analysis.
# Login-node OOMs on the accumulation lists; needs a compute node.
#
# Usage: sbatch run_perpos_H1024.sh

#SBATCH --job-name=perpos_h1024
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/perpos_h1024_%j.out
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
echo "Per-position dump for H=1024 wheneven"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python compare_aggregators.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 1024 \
    --save-per-position logs/perpos_H1024_wheneven.npz

echo "Completed: $(date)"
