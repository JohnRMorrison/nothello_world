#!/bin/bash
# Board-state probing on multi-seed MLP hidden layers.
#
# Usage:
#   sbatch probe_multi_seed_hidden.sh
#   NUM_TRAIN_GAMES=1000 MULTI_CKPT=<path> sbatch probe_multi_seed_hidden.sh

#SBATCH --job-name=probe_ms
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_ms_%j.out
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
VARIANTS=${VARIANTS:-all}

echo "============================================"
echo "Multi-seed hidden-state probing"
echo "  MULTI_CKPT=$MULTI_CKPT"
echo "  NUM_TRAIN_GAMES=$NUM_TRAIN_GAMES  NUM_TEST_GAMES=$NUM_TEST_GAMES"
echo "  EPOCHS=$EPOCHS"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

NSU_ARG=""
if [ -n "$NUM_SEEDS_USED" ]; then
    NSU_ARG="--num-seeds-used $NUM_SEEDS_USED"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_multi_seed_hidden.py \
    --multi-ckpt "$MULTI_CKPT" \
    --num-train-games $NUM_TRAIN_GAMES \
    --num-test-games $NUM_TEST_GAMES \
    --epochs $EPOCHS \
    --variants $VARIANTS \
    $NSU_ARG

echo "Completed at: $(date)"
