#!/bin/bash
#SBATCH --job-name=train_probes_L0_L3
#SBATCH --output=logs/train_probes_L0_L3_%j.out
#SBATCH --account=nklab
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -c 4
#SBATCH --time=08:00:00

source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

cd /engram/nklab/jrm2182/nothello_world
mkdir -p logs "Intervention Results"

for L in 0 1 2 3; do
    echo "===== training probe for block $L ====="
    python -u train_nanda_probe_extended.py \
        --ckpt ckpts/gpt_nanda_synthetic.ckpt \
        --layer $L \
        --output mechanistic_interpretability/main_linear_probe_L$L.pth 2>&1
done

python -u check_probe_accuracy.py --n-games 2000 \
    --probe mechanistic_interpretability/main_linear_probe_L0.pth:0 \
    --probe mechanistic_interpretability/main_linear_probe_L1.pth:1 \
    --probe mechanistic_interpretability/main_linear_probe_L2.pth:2 \
    --probe mechanistic_interpretability/main_linear_probe_L3.pth:3 \
    --probe mechanistic_interpretability/main_linear_probe_L4.pth:4 \
    --probe mechanistic_interpretability/main_linear_probe_L5.pth:5 \
    --probe mechanistic_interpretability/main_linear_probe.pth:6 \
    --probe mechanistic_interpretability/main_linear_probe_L7.pth:7 \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --out "Intervention Results/probe_accuracy_all_layers.txt" 2>&1
