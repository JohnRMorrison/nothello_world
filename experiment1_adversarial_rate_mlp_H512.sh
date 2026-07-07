#!/bin/bash
#SBATCH --job-name=exp1_mlp_H512
#SBATCH --output=logs/exp1_mlp_H512_%A_%a.out
#SBATCH --error=logs/exp1_mlp_H512_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --array=0-59

echo "Cell: ${SLURM_ARRAY_TASK_ID}"
echo "Started at: $(date)"

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

BASE=./experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

python -u experiment1_adversarial_rate_mlp.py \
    --cell ${SLURM_ARRAY_TASK_ID} \
    --mlp-ckpt $BASE/pattern_simple_direct_H512_playedeven.pt \
    --hidden 512 \
    --output-dir experiment1_data_mlp_H512 \
    --beam-width 10 --prefix-len 5 --max-depth 40 \
    --top-save 5

echo "Finished at: $(date)"
