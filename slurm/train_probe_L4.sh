#!/bin/bash
#SBATCH --job-name=train_probe_L4
#SBATCH --output=logs/train_probe_L4_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -c 4
#SBATCH --time=02:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world

mkdir -p logs

python train_nanda_probe_extended.py --layer 4 \
    --output mechanistic_interpretability/main_linear_probe_L4.pth
