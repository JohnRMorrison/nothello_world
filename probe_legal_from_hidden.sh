#!/bin/bash
# Legal-move probing from multi-seed hidden layers.
#
# Usage:
#   sbatch probe_legal_from_hidden.sh
#   MULTI_CKPT=<path> NUM_TRAIN_GAMES=10000 sbatch probe_legal_from_hidden.sh

#SBATCH --job-name=probe_legal
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_legal_%j.out
#SBATCH --account=nklab
#SBATCH --partition=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

MULTI_CKPT=${MULTI_CKPT:-experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/multi_seed_N100_H512_playedeven.pt}
NUM_TRAIN_GAMES=${NUM_TRAIN_GAMES:-5000}
NUM_TEST_GAMES=${NUM_TEST_GAMES:-500}
EPOCHS=${EPOCHS:-5}
NUM_SEEDS_USED=${NUM_SEEDS_USED:-}
VARIANTS=${VARIANTS:-concat,shared,moe}

echo "============================================"
echo "Legal-move probing from multi-seed hidden layers"
echo "  MULTI_CKPT=$MULTI_CKPT"
echo "  NUM_TRAIN_GAMES=$NUM_TRAIN_GAMES  NUM_TEST_GAMES=$NUM_TEST_GAMES"
echo "  EPOCHS=$EPOCHS  VARIANTS=$VARIANTS"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

NSU_ARG=""
if [ -n "$NUM_SEEDS_USED" ]; then
    NSU_ARG="--num-seeds-used $NUM_SEEDS_USED"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_legal_from_hidden.py \
    --multi-ckpt "$MULTI_CKPT" \
    --num-train-games $NUM_TRAIN_GAMES \
    --num-test-games $NUM_TEST_GAMES \
    --epochs $EPOCHS \
    --variants $VARIANTS \
    $NSU_ARG

echo "Completed at: $(date)"
