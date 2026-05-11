#!/bin/bash
#SBATCH --job-name=ogpt_intv
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ogpt_intv_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

LAYER=${1:-4}

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python ogpt_intervention.py \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --layer "$LAYER" \
    --scale 3 \
    --n-games 1000 \
    --test-positions-per-game 5 \
    --max-cells-per-pos 5 \
    --output "logs/ogpt_intervention_L${LAYER}.npz"
