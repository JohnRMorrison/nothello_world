#!/bin/bash
# Re-train selected per-turn probes for 10 epochs to address the
# "specialists trained on fewer examples" concern. 4 turns x 2 models
# (specialist + unified) = 8 jobs.
#
# Turns chosen to span the dynamics:
#   5  (specialist WINS +3pp on center cells)
#   25 (mid-game, modest unified advantage)
#   35 (mid-late, growing gap)
#   45 (late game, largest gap, -3.5pp on center)
#
# Usage: sbatch --array=0-7 exp_per_turn_probes_10ep.sh

#SBATCH --job-name=per_t10
#SBATCH -c 4
#SBATCH --time=03:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/per_t10_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TURNS=(5 25 35 45)
IDX=$SLURM_ARRAY_TASK_ID
TURN_IDX=$((IDX % 4))
T=${TURNS[$TURN_IDX]}

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

if [ $IDX -lt 4 ]; then
    CKPT=$CKPT_DIR/pattern_simple_direct_H512_wheneven_turn${T}.pt
    LABEL="specialist"
else
    CKPT=$CKPT_DIR/pattern_simple_direct_H512_wheneven.pt
    LABEL="unified"
fi

echo "10-epoch probe: model=$LABEL  turn=$T  ckpt=$CKPT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_per_turn_probe.py \
    --ckpt "$CKPT" --hidden 512 --target-turn "$T" --epochs 10
