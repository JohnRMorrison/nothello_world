#!/bin/bash
# Run compare_aggregators.py on the three pattern detector checkpoints from
# the recall@K experiments:
#   1. H=1024 + when+even          (capacity bump)
#   2. H=512 + when+even + listwise from scratch (lw=1.0)
#   3. finetune_for_legal --loss listwise on the wheneven baseline
#
# Each invocation prints a top-1 / top-3 / top-5 / top-10 table across all
# aggregators (max, mean, median, logsumexp, topk_mean-{2,3,5}, prob_or,
# count_pos, sum_sigmoid, ensemble_z).
#
# Usage: sbatch compare_aggregators_new_ckpts.sh

#SBATCH --job-name=cmp_aggr
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cmp_aggr_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

CKPT_H1024=$CKPT_DIR/pattern_simple_direct_H1024_wheneven.pt
CKPT_LISTW=$CKPT_DIR/pattern_simple_direct_H512_wheneven_listw1.pt
CKPT_FTLEG=$CKPT_DIR/ftlegal_pattern_simple_direct_H512_wheneven_output_listw.pt

echo "============================================"
echo "compare_aggregators on three new checkpoints"
echo "Job: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "============================================"

for CKPT in "$CKPT_H1024" "$CKPT_LISTW" "$CKPT_FTLEG"; do
    echo
    echo "--------------------------------------------"
    echo "Checkpoint: $CKPT"
    echo "--------------------------------------------"
    if [ ! -f "$CKPT" ]; then
        echo "  MISSING — skipping"
        continue
    fi
    # H is encoded in the filename; pull it out (H512 or H1024)
    H=$(echo "$CKPT" | grep -oE 'H[0-9]+' | head -1 | sed 's/H//')
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python compare_aggregators.py \
        --ckpt "$CKPT" \
        --mode direct \
        --hidden "$H"
done

echo
echo "Completed: $(date)"
