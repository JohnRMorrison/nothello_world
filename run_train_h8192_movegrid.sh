#!/bin/bash
# Train H=8192 movegrid pattern detector (3 epochs).
# ~37M params first layer + ~8M second layer = ~45M total.
# Note: train_pattern_simple.py handles the per-batch move_grid expansion.

#SBATCH --job-name=pat_h8k
#SBATCH -c 4
#SBATCH --time=18:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h8k_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --hidden 8192 --mode direct --epochs 3 --features move_grid \
    --loss bce
