#!/bin/bash
#SBATCH --job-name=beh_adv
#SBATCH --output=logs/beh_adv_%A_%a.out
#SBATCH --error=logs/beh_adv_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-59

echo "Cell: ${SLURM_ARRAY_TASK_ID}"
echo "Started at: $(date)"

source activate othello

python behavioral_adversarial.py \
    --cell ${SLURM_ARRAY_TASK_ID} \
    --output-dir behavioral_data \
    --ckpt ckpts/gpt_synthetic.ckpt \
    --num-starts 200 \
    --beam-width 20

echo "Finished at: $(date)"
