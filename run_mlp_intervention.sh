#!/bin/bash
#SBATCH --job-name=mlp_intv
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_intv_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H1024_wheneven.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python mlp_intervention.py \
    --ckpt "$CKPT" --hidden 1024 \
    --scales 1,2,3,5,8 \
    --n-train-probe 20000 --n-test 5000 \
    --output logs/mlp_intervention.npz
