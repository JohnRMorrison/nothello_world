#!/bin/bash
# Train a second MLP H=512 played+even with a different random seed, so we
# can compare its mistakes against the existing (seed=0) one.
#
# Usage:
#   SEED=42 sbatch train_mlp_h512_playedeven_seed.sh
#   SEED=1234 EPOCHS=3 sbatch train_mlp_h512_playedeven_seed.sh
#
# After completion, both checkpoints live in pattern_detector_checkpoints/
# with seed in the filename:
#   pattern_simple_direct_H512_playedeven_seed0.pt    (existing, renamed)
#   pattern_simple_direct_H512_playedeven_seed42.pt   (new)

#SBATCH --job-name=mlp_h512_seed
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_h512_seed_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

SEED=${SEED:?Must set SEED, e.g. SEED=42 sbatch train_mlp_h512_playedeven_seed.sh}
EPOCHS=${EPOCHS:-3}
CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints
BASE_NAME=pattern_simple_direct_H512_playedeven
DEFAULT_CKPT=$CKPT_DIR/$BASE_NAME.pt
TAGGED_NEW=$CKPT_DIR/${BASE_NAME}_seed${SEED}.pt
TAGGED_SEED0=$CKPT_DIR/${BASE_NAME}_seed0.pt

echo "============================================"
echo "Training MLP H=512 played+even, SEED=$SEED, EPOCHS=$EPOCHS"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

# Preserve the existing seed-0 checkpoint by renaming it (only first time).
if [ -f "$DEFAULT_CKPT" ] && [ ! -f "$TAGGED_SEED0" ]; then
    echo "Renaming existing default ckpt -> ${BASE_NAME}_seed0.pt for safekeeping."
    mv "$DEFAULT_CKPT" "$TAGGED_SEED0"
fi

# Train.  The script writes to $DEFAULT_CKPT regardless of seed.
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --mode direct \
    --hidden 512 \
    --features played+even \
    --epochs $EPOCHS \
    --chunk-prefix chunk_ext_ \
    --seed $SEED

# Move the newly-written ckpt to a seed-tagged filename so it doesn't clobber
# future runs.
if [ -f "$DEFAULT_CKPT" ]; then
    echo "Renaming new ckpt -> ${BASE_NAME}_seed${SEED}.pt"
    mv "$DEFAULT_CKPT" "$TAGGED_NEW"
fi

echo "Completed at: $(date)"
echo "Both checkpoints:"
ls -lh $CKPT_DIR/${BASE_NAME}_seed*.pt 2>/dev/null
