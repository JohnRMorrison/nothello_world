#!/bin/bash
#SBATCH --job-name=incoh_rules
#SBATCH --output=logs/incoh_rules_%j.out
#SBATCH --error=logs/incoh_rules_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Usage: sbatch incoherent_rules_experiment.sh <variant> [n_rules]
#   <variant> in {coherent, incoherent, proximal_nonlinear, distal_linear}
#   n_rules defaults to 100

VARIANT=${1:-coherent}
N_RULES=${2:-100}

source activate othello

echo "Variant: $VARIANT, n_rules: $N_RULES"
echo "Started at: $(date)"

python incoherent_rules_experiment.py \
    --variant "$VARIANT" \
    --n-rules "$N_RULES" \
    --output-dir experiments/incoherent_rules \
    --n-train 2000000 \
    --lr 5e-5 \
    --seed 42

echo "Finished at: $(date)"
