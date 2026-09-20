#!/bin/bash
#SBATCH --job-name=reeval_j1b_table
#SBATCH --output=logs/reeval_j1b_table_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=04:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world

mkdir -p logs "Intervention Results"

python reeval_argmax_legality.py \
  --probe-ckpts stream_out/J1_B.pt \
  --chunk-path experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_ext_0039.npz \
  --max-positions 500000 \
  --ply-min 5 \
  --ply-max 53
