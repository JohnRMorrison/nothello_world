#!/bin/bash
#SBATCH --job-name=gen_seqs
#SBATCH --output=logs/gen_seqs_%j.out
#SBATCH --error=logs/gen_seqs_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

echo "Started at: $(date)"

source activate othello

python mechanistic_interpretability/generate_board_seqs.py

echo "Finished at: $(date)"
