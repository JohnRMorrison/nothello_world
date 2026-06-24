#!/bin/bash
# Submit a chain of dependent v4 training jobs.  Each job starts after
# the previous one finishes (regardless of success), so SLURM time-limit
# kills don't break the chain.  Each chained job loads the most recent
# v4 checkpoint and trains for another EPOCHS epochs (default 20).
#
# Usage:
#   ./chain_v4_training.sh SEED_JOBID [N_CHAIN_JOBS]
#
# Example: continue from currently-running job 6009133 for 12 more
# 20-epoch jobs (gives ~250 epochs total assuming each job completes):
#   ./chain_v4_training.sh 6009133 12
#
# Notes:
#   - Cancel the chain anytime with: scancel <one of the job ids>
#     (cancelling a job only kills that one; the dependency on a
#     cancelled job is treated as 'afterany' so the chain continues
#     unless you cancel all subsequent ones too).
#   - To break the chain at a specific point, scancel each subsequent
#     job id individually.

set -e

SEED_JOB=${1:?Usage: chain_v4_training.sh SEED_JOBID [N_CHAIN_JOBS]}
N_CHAIN=${2:-12}
# Chained jobs use a constant low LR instead of restarting warmup+decay
# each time.  Default 5e-5 = end-of-cosine-decay LR from the seed schedule.
CONSTANT_LR=${CONSTANT_LR:-5e-5}

echo "Chaining $N_CHAIN jobs after seed job $SEED_JOB."
echo "Chained jobs will train at constant LR = $CONSTANT_LR (override via CONSTANT_LR env var)."
echo

JOB_PREV=$SEED_JOB
for i in $(seq 1 "$N_CHAIN"); do
    JOB_NEW=$(sbatch --parsable --dependency=afterany:"$JOB_PREV" \
                     --export=ALL,CONSTANT_LR="$CONSTANT_LR" \
                     train_gpt_shuffled_v4.sh)
    echo "  Chain step $i: job $JOB_NEW  (after $JOB_PREV)"
    JOB_PREV=$JOB_NEW
done

echo
echo "Done.  Run 'squeue -u \$USER' to view the chain."
