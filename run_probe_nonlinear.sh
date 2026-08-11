#!/bin/bash
# Train FULL-CHUNK non-linear board probes (NonLinearProbe: hidden->512->64x3)
# on the hidden layer of the 4 pattern-detector MLPs, matching how the LINEAR
# probe_direct_* probes were trained (same probe_pattern_models.py, same data,
# same --epochs 5, default chunk_ prefix). This makes the linear-vs-nonlinear
# board-decode comparison fair (the earlier pod-trained nonlinear was a 200k
# undertrained floor).
#
# Array task -> MLP:
#   0: H512_playedeven   1: H512_move_grid
#   2: H4096_playedeven  3: H4096_move_grid
#
# Usage: sbatch run_probe_nonlinear.sh

#SBATCH --job-name=probe_nl
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/probe_nl_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKDIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints
SPECS=(H512_playedeven H512_move_grid H4096_playedeven H4096_move_grid)
SPEC=${SPECS[$SLURM_ARRAY_TASK_ID]}
HID=${SPEC%%_*}          # H512 / H4096
HID=${HID#H}             # 512 / 4096

echo "Task $SLURM_ARRAY_TASK_ID -> $SPEC (hidden=$HID)"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python probe_pattern_models.py \
    --ckpt "$CKDIR/pattern_simple_direct_${SPEC}.pt" \
    --mode direct --hidden "$HID" --nonlinear --epochs 5
