#!/bin/bash
#SBATCH --job-name=intv_alpha
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/intv_alpha_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs experiments/intervention_alpha_sweep
cd $SLURM_SUBMIT_DIR

# Optional positional args:
#   $1 = layer (default 6)
#   $2 = n_games (default 500)
#   $3 = mode (default empty): 'empty' or 'flip'
LAYER=${1:-6}
N_GAMES=${2:-500}
MODE=${3:-empty}
OUT_DIR="experiments/intervention_alpha_sweep"
TAG="L${LAYER}_n${N_GAMES}_${MODE}"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python sweep_intervention_alpha.py \
    --ckpt ckpts/gpt_nanda_synthetic.ckpt \
    --probe mechanistic_interpretability/main_linear_probe.pth \
    --layer "$LAYER" \
    --n-games "$N_GAMES" \
    --mode "$MODE" \
    --alphas 0,1,2,4,6,8,12,16,24 \
    --squares 3,3 0,3 2,3 \
    --output "${OUT_DIR}/results_${TAG}.json" \
    --raw-npz "${OUT_DIR}/raw_flips_${TAG}.npz"

# Plot inline so the PDFs land next to the JSON without a separate job.
python plot_intervention_alpha_vs_errors.py \
    --results "${OUT_DIR}/results_${TAG}.json" \
    --out     "${OUT_DIR}/alpha_vs_errors_${TAG}.pdf"

python plot_intervention_alpha_cellmaps.py \
    --raw     "${OUT_DIR}/raw_flips_${TAG}.npz" \
    --results "${OUT_DIR}/results_${TAG}.json" \
    --out     "${OUT_DIR}/cellmaps_${TAG}.pdf"
