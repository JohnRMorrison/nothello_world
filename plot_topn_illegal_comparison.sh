#!/bin/bash
# Generate the MLP and OGPT "top-1 prediction is illegal" figures on the
# same set of games.
#
# Usage:
#   sbatch plot_topn_illegal_comparison.sh
#   NUM_GAMES=100000 sbatch plot_topn_illegal_comparison.sh
#   MLP_CKPT=<path> sbatch plot_topn_illegal_comparison.sh

#SBATCH --job-name=topn_illegal
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/topn_illegal_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs experiments/plots
cd $SLURM_SUBMIT_DIR

NUM_GAMES=${NUM_GAMES:-100000}
MLP_CKPT=${MLP_CKPT:-experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_playedeven_seed0.pt}
MLP_HIDDEN=${MLP_HIDDEN:-512}
OGPT_CKPT=${OGPT_CKPT:-ckpts/gpt_nanda_synthetic.ckpt}
NUM_DATA_FILES=${NUM_DATA_FILES:-2}

echo "============================================"
echo "Top-1 illegal comparison (MLP vs OGPT)"
echo "  NUM_GAMES=$NUM_GAMES"
echo "  MLP_CKPT=$MLP_CKPT  (H=$MLP_HIDDEN)"
echo "  OGPT_CKPT=$OGPT_CKPT"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started at: $(date)"
echo "============================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python plot_topn_illegal_comparison.py \
    --mlp-ckpt "$MLP_CKPT" \
    --mlp-hidden $MLP_HIDDEN \
    --ogpt-ckpt "$OGPT_CKPT" \
    --num-games $NUM_GAMES \
    --num-data-files $NUM_DATA_FILES

echo "Completed at: $(date)"
