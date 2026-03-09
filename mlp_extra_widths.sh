#!/bin/bash
#SBATCH --job-name=mlp_xtra
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_xtra_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SLURM_SUBMIT_DIR

HIDDEN_ARRAY=(2048 4096 8192 16384)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
H=${HIDDEN_ARRAY[$TASK_ID]}

echo "Training MLP H=${H} on 1000000 games, 10 epochs"

CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
    --experiment mlp \
    --precomputed \
    --mlp-hidden $H \
    --mlp-only \
    --max-games 1000000 \
    --epochs 10 \
    --output-dir experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "Completed at: $(date)"
