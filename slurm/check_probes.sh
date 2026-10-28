#!/bin/bash
#SBATCH --job-name=check_probes
#SBATCH --output=logs/check_probes_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH --time=01:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
mkdir -p logs "Intervention Results"

python -u check_probe_accuracy.py --n-games 2000 \
    --out "Intervention Results/probe_accuracy.txt" 2>&1
