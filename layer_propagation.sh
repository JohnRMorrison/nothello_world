#!/bin/bash
#SBATCH --job-name=layer_prop
#SBATCH --output=logs/layer_prop_%A_%a.out
#SBATCH --error=logs/layer_prop_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=4-6

echo "Started at: $(date)"
echo "Layer: ${SLURM_ARRAY_TASK_ID}"

source activate othello

cd mechanistic_interpretability

python layer_propagation.py \
    --layer ${SLURM_ARRAY_TASK_ID} \
    --probe-dir ../experiments/mathematical_transformation_experiments/heuristic_probe_results/probe_directions/probe_checkpoints \
    --cal-depths 0,1,2 \
    --n-games 200 \
    --output-dir ../experiments/layer_propagation/L${SLURM_ARRAY_TASK_ID} \
    --seed 42

echo "Finished at: $(date)"
