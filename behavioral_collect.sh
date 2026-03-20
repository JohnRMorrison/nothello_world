#!/bin/bash
#SBATCH --job-name=beh_coll
#SBATCH --output=logs/beh_collect_%A_%a.out
#SBATCH --error=logs/beh_collect_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-19

echo "Shard: ${SLURM_ARRAY_TASK_ID}"
echo "Started at: $(date)"

source activate othello

python behavioral_collect.py \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --output-dir behavioral_data \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --num-games 100000

echo "Finished at: $(date)"
