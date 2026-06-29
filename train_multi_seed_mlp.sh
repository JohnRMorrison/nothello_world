#!/bin/bash
# Train N MLPs simultaneously with shared data loading.

#SBATCH --job-name=multi_seed
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/multi_seed_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

HIDDEN=${HIDDEN:-512}
NUM_SEEDS=${NUM_SEEDS:-100}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8192}
SEED=${SEED:-0}
MAX_CHUNKS=${MAX_CHUNKS:-}

echo "============================================"
echo "Multi-seed MLP training"
echo "  H=$HIDDEN  N=$NUM_SEEDS  EPOCHS=$EPOCHS  BATCH=$BATCH_SIZE  SEED=$SEED"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

MAX_CHUNKS_ARG=""
if [ -n "$MAX_CHUNKS" ]; then
    MAX_CHUNKS_ARG="--max-chunks $MAX_CHUNKS"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_multi_seed_mlp.py \
    --num-seeds $NUM_SEEDS \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    $MAX_CHUNKS_ARG

echo "Completed at: $(date)"
