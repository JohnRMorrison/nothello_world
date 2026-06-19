#!/bin/bash
# Run minimal-circuits analysis on ONE (pattern, parity) instance.
#
# Configure these three at the top, then sbatch this script:
PATTERN_IDX=${PATTERN_IDX:-0}
PARITY=${PARITY:-even}
MAX_K=${MAX_K:-6}
FEATURES=${FEATURES:-playedeven}
CKPT_NAME=${CKPT_NAME:-pattern_simple_direct_H512_playedeven.pt}
#
# Usage:
#   sbatch --export=PATTERN_IDX=42,PARITY=odd,MAX_K=6 run_minimal_circuits_single.sh
# Or change the defaults above and just `sbatch run_minimal_circuits_single.sh`.

#SBATCH --job-name=mincirc1
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --output=logs/mincirc_p%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
mkdir -p minimal_circuits_results
cd $SLURM_SUBMIT_DIR

CKPT=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/${CKPT_NAME}
OUT=minimal_circuits_results/p${PATTERN_IDX}_${PARITY}_${FEATURES}_K${MAX_K}.pkl.gz

echo "Running pattern_idx=${PATTERN_IDX}, parity=${PARITY}, features=${FEATURES}, max_K=${MAX_K}"
echo "Checkpoint: ${CKPT}"
echo "Output:     ${OUT}"
echo ""

PYTHONUNBUFFERED=1 python enumerate_minimal_circuits.py \
    --ckpt $CKPT \
    --features $FEATURES \
    --pattern-idx $PATTERN_IDX \
    --parity $PARITY \
    --max-k $MAX_K \
    --output $OUT \
    --verbose

echo ""
echo "Done. Output at: ${OUT}"
echo "Inspect with:"
echo "  python -c \"import pickle, gzip; d = pickle.load(gzip.open('${OUT}', 'rb')); print(d['status'], 'minimal_input_sets:', len(d['minimal_input_sets']), 'by_size:', d.get('minimal_input_by_size'))\""
