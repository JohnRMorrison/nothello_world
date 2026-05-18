#!/bin/bash
# Re-train Nanda-style linear probe on OGPT, covering positions 5-58
# instead of his original 5-53. Faithful to tl_probing_v1.py but via
# mingpt forward pass.
#
# Output: mechanistic_interpretability/main_linear_probe_ext_5_58.pth

#SBATCH --job-name=nanda_ext
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/nanda_ext_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_nanda_probe_extended.py \
    --epochs 2 --n-games 100000 --layer 6 \
    --pos-start 5 --pos-end 59
