#!/bin/bash
# Precompute feature chunks for 12M games (chunks 20-39).
# Chunks 0-19 already exist (6M games). This adds another 6M.
#
# Usage: sbatch --array=20-39 precompute_12m.sh
#
# Each chunk: 3 files × ~100K games = ~300K games → ~14.7M samples.
# Total with existing: 40 chunks = 120 files = ~12M games.

#SBATCH --job-name=precomp12m
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --output=logs/precompute_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

FILES_PER_CHUNK=3
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-20}
FILE_START=$((CHUNK_ID * FILES_PER_CHUNK))
FILE_END=$((FILE_START + FILES_PER_CHUNK))

OUTPUT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results

echo "============================================"
echo "Precompute chunk $CHUNK_ID (files $FILE_START-$((FILE_END-1)))"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $CHUNK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python -m experiments.mathematical_transformation_experiments.heuristic_probe_experiments \
    --experiment precompute \
    --chunk-id $CHUNK_ID \
    --file-start $FILE_START \
    --file-end $FILE_END \
    --output-dir $OUTPUT_DIR

echo "Completed at: $(date)"
