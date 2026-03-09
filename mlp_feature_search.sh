#!/bin/bash
# ============================================================================
# Train MLP on move-history features (180-d) with large game datasets
# ============================================================================
#
# Two-step workflow:
#
#   Step 1: Precompute features in parallel (one job per file chunk)
#     sbatch --array=0-3 mlp_feature_search.sh precompute
#     (4 chunks × 3 files each = 12 files, ~1.2M games)
#
#   Step 2: Train MLPs (width sweep), one per array task
#     sbatch --array=0-8 mlp_feature_search.sh train
#     (trains H=4,8,16,32,64,128,256,512,1024 — one per task)
#
#   Step 3: Dropout evaluation on trained H=1024 MLP
#     sbatch mlp_feature_search.sh dropout
#
# Or run single-job (small scale):
#     sbatch mlp_feature_search.sh
# ============================================================================

#SBATCH --job-name=mlp_feat
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_feat_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

# Load CUDA
module load cuda/11.8.0

# Activate conda environment
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs

cd $SLURM_SUBMIT_DIR

# Parse arguments
FILES_PER_CHUNK=3
MODE="single"  # precompute, train, dropout, or single
MAX_GAMES=1000000
EPOCHS=10
for arg in "$@"; do
    if [[ "$arg" == chunk=* ]]; then
        FILES_PER_CHUNK="${arg#chunk=}"
    elif [[ "$arg" == games=* ]]; then
        MAX_GAMES="${arg#games=}"
    elif [[ "$arg" == epochs=* ]]; then
        EPOCHS="${arg#epochs=}"
    elif [[ "$arg" == "precompute" ]]; then
        MODE="precompute"
    elif [[ "$arg" == "train" ]]; then
        MODE="train"
    elif [[ "$arg" == "dropout" ]]; then
        MODE="dropout"
    fi
done

# Hidden dims indexed by array task ID
HIDDEN_ARRAY=(4 8 16 32 64 128 256 512 1024)

OUTPUT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "============================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Mode: ${MODE}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

if [[ "$MODE" == "precompute" ]]; then
    # Parallel precompute: each array task handles FILES_PER_CHUNK files
    CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}
    FILE_START=$((CHUNK_ID * FILES_PER_CHUNK))
    FILE_END=$((FILE_START + FILES_PER_CHUNK))

    echo "Chunk ${CHUNK_ID}: files ${FILE_START}-$((FILE_END - 1))"

    python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
        --experiment precompute \
        --chunk-id $CHUNK_ID \
        --file-start $FILE_START \
        --file-end $FILE_END \
        --output-dir $OUTPUT_DIR

elif [[ "$MODE" == "train" ]]; then
    # Each array task trains one hidden dim
    TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
    H=${HIDDEN_ARRAY[$TASK_ID]}
    echo "Training MLP H=${H} on ${MAX_GAMES} games, ${EPOCHS} epochs"

    CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
        --experiment mlp \
        --precomputed \
        --mlp-hidden $H \
        --mlp-only \
        --max-games $MAX_GAMES \
        --epochs $EPOCHS \
        --output-dir $OUTPUT_DIR

elif [[ "$MODE" == "dropout" ]]; then
    # Dropout evaluation on trained H=1024 MLP
    echo "Dropout evaluation"
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
        --experiment mlp_dropout \
        --precomputed \
        --mlp-hidden 1024 \
        --max-games $MAX_GAMES \
        --output-dir $OUTPUT_DIR

else
    # Single-job mode (small scale, no precompute)
    echo "Single job: files_per_chunk=${FILES_PER_CHUNK}"
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
        --experiment mlp \
        --max-files $FILES_PER_CHUNK \
        --max-games $MAX_GAMES \
        --mlp-hidden ${HIDDEN_ARRAY[@]} \
        --mlp-only \
        --epochs $EPOCHS \
        --output-dir $OUTPUT_DIR
fi

echo "Completed at: $(date)"
