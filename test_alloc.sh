#!/bin/bash
# ============================================================================
# Minimal scheduling probe.  Requests almost nothing so it should backfill
# and start quickly IF the queue is functioning for you.  Override --mem /
# --gres on the sbatch command line to isolate what is blocking your real
# jobs (see the three test commands below).
#
#   sbatch test_alloc.sh                          # baseline: no GPU, 2G
#   sbatch --mem=240G test_alloc.sh               # memory alone
#   sbatch --gres=gpu:1 --mem=4G test_alloc.sh    # GPU alone
# ============================================================================

#SBATCH --job-name=alloc_test
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --account=nklab
#SBATCH --output=logs/alloc_test_%j.out

mkdir -p logs
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Partition:   ${SLURM_JOB_PARTITION}"
echo "CPUs:        ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:    ${SLURM_MEM_PER_NODE} MB"
echo "GPUs:        ${SLURM_GPUS:-none} (${CUDA_VISIBLE_DEVICES:-unset})"
echo "Started at:  $(date)"
echo "============================================"
sleep 60
echo "Completed at: $(date)"
