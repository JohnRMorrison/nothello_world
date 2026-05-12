#!/bin/bash
# Precompute forfeit-correct by_black channel for all feature chunks.
# One-time, ~30-60 min depending on chunk count.
#
# Usage: sbatch run_precompute_by_black.sh

#SBATCH --job-name=by_black
#SBATCH -c 32
#SBATCH --time=12:00:00
#SBATCH --mem=120GB
#SBATCH --output=logs/by_black_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 python precompute_by_black.py --workers 32
