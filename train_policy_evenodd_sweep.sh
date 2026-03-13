#!/bin/bash
# Sweep even/odd policy heads: H={1024,2048} x pos_weight={1.0,1.5,2.0} + 2-layer H=1024
# Each job trains BOTH even and odd heads sequentially
# Tasks 0-2:  H=1024 L=1, Tasks 3-5: H=2048 L=1, Tasks 6-8: H=1024 L=2
# Usage: sbatch --array=0-8 train_policy_evenodd_sweep.sh

#SBATCH --job-name=eo_sweep
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/policy_eo_sweep_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p logs
cd $SLURM_SUBMIT_DIR

PW_ARRAY=(1.0 1.5 2.0)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Tasks 0-2: H=1024 L=1, Tasks 3-5: H=2048 L=1, Tasks 6-8: H=1024 L=2
if [ $TASK_ID -lt 3 ]; then
    HIDDEN=1024
    LAYERS=1
    PW_IDX=$TASK_ID
elif [ $TASK_ID -lt 6 ]; then
    HIDDEN=2048
    LAYERS=1
    PW_IDX=$((TASK_ID - 3))
else
    HIDDEN=1024
    LAYERS=2
    PW_IDX=$((TASK_ID - 6))
fi
PW=${PW_ARRAY[$PW_IDX]}

echo "============================================"
echo "Even/odd policy head sweep: H=$HIDDEN, L=$LAYERS, pos_weight=$PW, 6M games"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}, Task: $TASK_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

python generate_adversarial_games.py \
    --train-policy \
    --max-games 6000000 \
    --policy-epochs 10 \
    --policy-hidden $HIDDEN \
    --policy-layers $LAYERS \
    --pos-weight $PW \
    --even-odd

echo "Completed at: $(date)"
