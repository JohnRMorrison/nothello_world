#!/bin/bash
#SBATCH --job-name=new_sq
#SBATCH --output=logs/new_sq_%A_%a.out
#SBATCH --error=logs/new_sq_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#
# New-squares transfer experiment (GPT).  Staleness-proofed by splitting the
# CPU game generation from the GPU training, so each piece is a short,
# backfill-friendly job that slots into any gap:
#
#   STAGE=data   generate the shared games + test manifest (CPU only; submit
#                with --gres=gpu:0 --mem=16G so it backfills anywhere)
#   STAGE=train  load the cached data, fine-tune + eval on GPU (tiny footprint)
#   STAGE=full   both in one job (default; simplest, but holds a GPU during gen)
#
# Condition (0=coherent, 1=incoherent) comes from the array task id.
#
# Recommended (stale-proof) flow -- data first on CPU, then train on GPU:
#   STAGE=data  sbatch --array=0-1 --gres=gpu:0 --mem=16G new_squares_experiment.sh
#   STAGE=train sbatch --array=0-1 new_squares_experiment.sh
# One-shot:
#   sbatch --array=0-1 new_squares_experiment.sh

source activate othello
cd $SLURM_SUBMIT_DIR

STAGE=${STAGE:-full}
COND=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEED:-42}
BUDGET=${BUDGET:-1000000}         # Control 1: new-square-legal positions per condition
EPOCHS=${EPOCHS:-3}               # passes over the games (toward convergence)
CKPT=${CKPT:-./ckpts/gpt_synthetic.ckpt}
OUT=${OUT:-experiments/new_squares}

echo "STAGE=$STAGE  condition=$COND  seed=$SEED  budget=$BUDGET  epochs=$EPOCHS  ckpt=$CKPT"
echo "Started at: $(date)"

case "$STAGE" in
  data)
    python new_squares_data.py \
        --condition-id $COND --output-dir $OUT \
        --new-square-legal-budget $BUDGET --n-test-positions 5000 --seed $SEED
    ;;
  train|full)
    # train auto-loads the cached data written by STAGE=data (regenerates only
    # if absent), so this is short on GPU.  Retention eval is ON (not skipped).
    python new_squares_transfer.py \
        --condition-id $COND --output-dir $OUT \
        --new-square-legal-budget $BUDGET --n-test-positions 5000 \
        --ckpt $CKPT --epochs $EPOCHS --lr 5e-5 --bs 16 --seed $SEED
    ;;
  *)
    echo "unknown STAGE '$STAGE' -- use data|train|full"; exit 1 ;;
esac

echo "Finished at: $(date)"
