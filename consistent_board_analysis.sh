#!/bin/bash
# Consistent-board analysis: legal-move + probe accuracy across all
# board states compatible with a moveset.

#SBATCH --job-name=consist_board
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/consist_board_%j.out
#SBATCH --account=nklab
#SBATCH --partition=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

MULTI_CKPT=${MULTI_CKPT:-experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/multi_seed_N100_H512_playedeven_chunks0-9.pt}
K=${K:-25}
NUM_GAMES=${NUM_GAMES:-1000}
N_SAMPLES=${N_SAMPLES:-1000}
PROBE_TRAIN_GAMES=${PROBE_TRAIN_GAMES:-2000}
PROBE_EPOCHS=${PROBE_EPOCHS:-5}
OUTPUT_CSV=${OUTPUT_CSV:-consistent_board_k${K}.csv}
SEED_IDX=${SEED_IDX:-}   # if set, use only that single MLP

echo "============================================"
echo "Consistent-board analysis"
echo "  MULTI_CKPT=$MULTI_CKPT"
echo "  K=$K  NUM_GAMES=$NUM_GAMES  N_SAMPLES=$N_SAMPLES"
echo "  PROBE_TRAIN_GAMES=$PROBE_TRAIN_GAMES  PROBE_EPOCHS=$PROBE_EPOCHS"
echo "  OUTPUT_CSV=$OUTPUT_CSV"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

SEED_IDX_ARG=""
if [ -n "$SEED_IDX" ]; then
    SEED_IDX_ARG="--seed-idx $SEED_IDX"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python consistent_board_analysis.py \
    --multi-ckpt "$MULTI_CKPT" \
    --k $K \
    --num-games $NUM_GAMES \
    --n-samples $N_SAMPLES \
    --probe-train-games $PROBE_TRAIN_GAMES \
    --probe-epochs $PROBE_EPOCHS \
    --output-csv $OUTPUT_CSV \
    $SEED_IDX_ARG

echo "Completed at: $(date)"
