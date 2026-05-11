#!/bin/bash
# Sanity check on the pattern-length finding: is the model failing because of
# pattern length itself, or because long patterns span flipped cells the
# move-history features can't track?
#
# Usage: sbatch run_pattern_flips.sh

#SBATCH --job-name=pat_flips
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_flips_%j.out
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
echo "Recall by (pattern length, flipped-cell count)"
echo "Checkpoint: $CKPT"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python pattern_flips_analysis.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 1024

echo "Completed: $(date)"
