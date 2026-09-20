#!/bin/bash
#SBATCH --job-name=alpha_sweep_L6
#SBATCH --output=logs/alpha_sweep_L6_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=08:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world

mkdir -p logs "Intervention Results"

python -u run_alpha_sweep.py \
    --intervention-layer 6 \
    --n-positions 200 \
    --out-suffix _L6
