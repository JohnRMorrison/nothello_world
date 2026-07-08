#!/bin/bash
#SBATCH --job-name=exp1_by_depth
#SBATCH --output=logs/exp1_by_depth_%A_%a.out
#SBATCH --error=logs/exp1_by_depth_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Array indexes 0..3 map to depths 10, 15, 20, 25
DEPTHS=(10 15 20 25)
DEPTH=${DEPTHS[${SLURM_ARRAY_TASK_ID}]}

echo "Depth: ${DEPTH}"
echo "Started at: $(date)"

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

python -u experiment1_adversarial_rate_by_depth.py \
    --depth ${DEPTH} \
    --n-samples 1000 \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --output-dir experiment1_by_depth \
    --beam-width 10 --max-depth 40

echo "Finished at: $(date)"
