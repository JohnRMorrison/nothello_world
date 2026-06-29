#!/bin/bash
# Train a Variant A (output-only) stacking readout on the 3-MLP ensemble.

#SBATCH --job-name=agg_readout
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/agg_readout_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints
CHUNK_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks

TRAIN_CHUNK=${TRAIN_CHUNK:-$CHUNK_DIR/chunk_ext_0037.npz}
TEST_CHUNK=${TEST_CHUNK:-$CHUNK_DIR/chunk_ext_0039.npz}
TRAIN_SIZE=${TRAIN_SIZE:-5000000}
TEST_SIZE=${TEST_SIZE:-5000000}
READOUT_HIDDEN=${READOUT_HIDDEN:-128}
READOUT_EPOCHS=${READOUT_EPOCHS:-10}

echo "============================================"
echo "Stacking readout: hidden=$READOUT_HIDDEN  epochs=$READOUT_EPOCHS"
echo "  train_chunk=$TRAIN_CHUNK  size=$TRAIN_SIZE"
echo "  test_chunk=$TEST_CHUNK    size=$TEST_SIZE"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

POS_FILTER_ARGS=""
if [ -n "$POS_MIN" ] && [ -n "$POS_MAX" ]; then
    POS_FILTER_ARGS="--pos-min $POS_MIN --pos-max $POS_MAX"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_aggregator_readout.py \
    --ckpts $CKPT_DIR/pattern_simple_direct_H512_playedeven_seed0.pt \
            $CKPT_DIR/pattern_simple_direct_H512_playedeven_seed43.pt \
            $CKPT_DIR/pattern_simple_direct_H512_playedeven_seed44.pt \
    --train-chunk $TRAIN_CHUNK \
    --test-chunk $TEST_CHUNK \
    --train-size $TRAIN_SIZE \
    --test-size $TEST_SIZE \
    --readout-hidden $READOUT_HIDDEN \
    --readout-epochs $READOUT_EPOCHS \
    $POS_FILTER_ARGS

echo "Completed at: $(date)"
