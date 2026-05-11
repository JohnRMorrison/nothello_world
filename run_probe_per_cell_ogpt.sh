#!/bin/bash
#SBATCH --job-name=probe_ogpt
#SBATCH -c 4
#SBATCH --time=01:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_ogpt_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_per_cell_ogpt.py \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --layers 2,4,6 \
    --n-games 5000 \
    --output logs/probe_per_cell_ogpt.npz
