#!/bin/bash
#SBATCH --job-name=topn_compare
#SBATCH --output=logs/topn_compare_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=04:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world

mkdir -p logs experiments

python topn_intervention_compare.py --n-positions 5000 \
    --models j1b,ogpt --categories flip --cal-depth 2 --K 4 \
    --out experiments/topn_compare_cd2_K4
