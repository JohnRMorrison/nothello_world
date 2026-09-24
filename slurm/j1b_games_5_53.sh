#!/bin/bash
#SBATCH --job-name=j1b_games_5_53
#SBATCH --output=logs/j1b_games_5_53_%j.out
#SBATCH --account=nklab
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH --time=04:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world

mkdir -p logs

# Per-GAME illegal mass over moves 5-53, on held-out games, so the number is
# comparable to the Othello-GPT column (which is a per-game max over
# positions).  The top-K FRAC line it also prints is the check: prob-OR top-1
# should land near the chunk path's ~98.5%.  If it does not, the features are
# wrong and the mass numbers mean nothing.
python eval_j1b_games.py \
  --probe-ckpts stream_out/J1_B.pt \
  --data-dir ./data/othello_synthetic \
  --num-data-files 3 \
  --num-games 8000 \
  --ply-min 5 \
  --ply-max 53 \
  --ks 1 3 5 \
  --out-npz j1b_illegal_5_53.npz
