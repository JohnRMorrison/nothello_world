#!/bin/bash
# compare_aggregators.py on the H=2048 wheneven checkpoint (2 epochs complete).
#
# Usage: sbatch run_compare_aggregators_H2048.sh

#SBATCH --job-name=cmp_h2048
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cmp_h2048_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H2048_wheneven.pt

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python compare_aggregators.py \
    --ckpt "$CKPT" \
    --mode direct --hidden 2048
