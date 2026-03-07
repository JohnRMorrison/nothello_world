#!/bin/bash
# ============================================================================
# Probe Othello-GPT for Othello board state — one job per layer
# ============================================================================

#SBATCH --job-name=probe_og
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/probe_og_%A_layer%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Load CUDA
module load cuda/11.8.0

# Activate conda environment
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs

cd $SLURM_SUBMIT_DIR

LAYER=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Probing layer: ${LAYER}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.probe_state_pred_for_othello \
    --ckpt-path ckpts/gpt_synthetic.ckpt \
    --layer $LAYER \
    --max-games 100000 \
    --probe-epochs 15 \
    --output-dir experiments/mathematical_transformation_experiments/othello_gpt_probe_results

echo "Completed at: $(date)"
