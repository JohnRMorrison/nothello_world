#!/bin/bash
# Per-cell turn-stratified analysis of the random-projection probe.
# Tests whether the ~94-95% probe accuracy reported for random projection
# is subject to the same first-490k eval bias (turns 5-6 only) as the
# trained MLP probes were.
#
# Usage:
#   sbatch run_randproj_per_cell.sh 512    # or 1024 / 2048 / 4096

#SBATCH --job-name=rp_pc
#SBATCH -c 2
#SBATCH --time=00:20:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rp_pc_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

H=${1:-512}
# The randproj checkpoint filename pattern includes seed and proj scale.
# Use the first one matching the H value.
CKPT=$(ls experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_randproj_*H${H}*.pt 2>/dev/null | head -1)
PROBE=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/probe_randproj_H${H}.pt

if [ ! -f "$CKPT" ] || [ ! -f "$PROBE" ]; then
    echo "Missing ckpt or probe (H=$H):"
    echo "  CKPT=$CKPT  exists=$([ -f $CKPT ] && echo yes || echo no)"
    echo "  PROBE=$PROBE  exists=$([ -f $PROBE ] && echo yes || echo no)"
    exit 1
fi

echo "Analyzing: CKPT=$CKPT, PROBE=$PROBE, H=$H"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python analyze_mlp_saved_probe_per_cell.py \
    --ckpt "$CKPT" --probe "$PROBE" --hidden "$H" \
    --features when
