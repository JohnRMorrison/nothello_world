#!/bin/bash
#SBATCH --job-name=alpha_sweep_L7
#SBATCH --output=logs/alpha_sweep_L7_%j.out
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
    --intervention-layer 7 \
    --probe-path mechanistic_interpretability/main_linear_probe_L7.pth \
    --n-positions 200 \
    --out-suffix _L7
