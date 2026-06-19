#!/bin/bash
#SBATCH --job-name=mincirc
#SBATCH -c 2
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --output=logs/mincirc_arr_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01
#
# Array job: one task per pattern index. Set parity via env var.
# Submit twice (once per parity) to cover all 1920 instances.
#
# Usage:
#   # Run pattern 0..959 with parity=even, throttled to 16 concurrent
#   sbatch --array=0-959%16 --export=PARITY=even,MAX_K=5 run_minimal_circuits_array.sh
#
#   # Then for odd parity
#   sbatch --array=0-959%16 --export=PARITY=odd,MAX_K=5 run_minimal_circuits_array.sh
#
# Throttle: change %16 to %8 (gentler) or %32 (faster).

PATTERN_IDX=${SLURM_ARRAY_TASK_ID}
PARITY=${PARITY:-even}
MAX_K=${MAX_K:-5}
FEATURES=${FEATURES:-playedeven}
CKPT_NAME=${CKPT_NAME:-pattern_simple_direct_H512_playedeven.pt}

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
mkdir -p minimal_circuits_results
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/${CKPT_NAME}
OUT=minimal_circuits_results/p${PATTERN_IDX}_${PARITY}_${FEATURES}_K${MAX_K}.pkl.gz

# Skip if already done (allows resubmitting array without redoing finished work)
if [ -f "$OUT" ]; then
    echo "Skipping: $OUT already exists"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] Pattern $PATTERN_IDX, parity=$PARITY, max_K=$MAX_K"

PYTHONUNBUFFERED=1 python enumerate_minimal_circuits.py \
    --ckpt $CKPT \
    --features $FEATURES \
    --pattern-idx $PATTERN_IDX \
    --parity $PARITY \
    --max-k $MAX_K \
    --output $OUT

echo "[$(date '+%H:%M:%S')] Done: $OUT"
