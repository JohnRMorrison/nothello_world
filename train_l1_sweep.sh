#!/bin/bash
#SBATCH --job-name=l1sweep
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/l1_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01
#
# Sweep over (feature_set, l1_weight) combinations:
#   array idx 0..4 → playedeven, L1 ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
#   array idx 5..9 → move_grid,  L1 ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
#
# Usage:
#   sbatch --array=0-9%4 train_l1_sweep.sh

L1_VALUES=(1e-4 3e-4 1e-3 3e-3 1e-2)
FEATURE_SETS=(playedeven move_grid)

# Map array index -> (feature, l1)
FEAT_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
L1_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
FEATURES=${FEATURE_SETS[$FEAT_IDX]}
L1=${L1_VALUES[$L1_IDX]}

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "Task ${SLURM_ARRAY_TASK_ID}: features=${FEATURES}, l1=${L1}"

# Map feature set to --features arg and --chunk-prefix
if [ "$FEATURES" = "playedeven" ]; then
    FEAT_ARG="played+even"
else
    FEAT_ARG="move_grid"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 512 \
    --features $FEAT_ARG \
    --epochs 3 \
    --chunk-prefix chunk_ext_ \
    --l1-weight $L1

echo "Done"
