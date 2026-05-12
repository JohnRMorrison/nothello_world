#!/bin/bash
#SBATCH --job-name=eval_spec
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_spec_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python eval_turn_specialists.py
