#!/bin/bash
#SBATCH --job-name=nanda_benchmark
#SBATCH --output=logs/nanda_benchmark_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH --time=03:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
mkdir -p logs "Intervention Results"

echo "node $(hostname)"
md5sum run_nanda_benchmark.py src/intervention.py

python -u run_nanda_benchmark.py \
    --out "Intervention Results/nanda_benchmark" 2>&1
