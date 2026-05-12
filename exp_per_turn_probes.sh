#!/bin/bash
# SLURM array: train per-turn Nanda-style probes for both specialist and
# unified MLPs at each of 10 target turns. 20 total probes, run in parallel.
#
# Array indices 0..9 → specialist probes (turns 5, 10, ..., 50)
# Array indices 10..19 → unified probes (turns 5, 10, ..., 50)
#
# Usage: sbatch --array=0-19 exp_per_turn_probes.sh

#SBATCH --job-name=per_t_probe
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/per_t_probe_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

TURNS=(5 10 15 20 25 30 35 40 45 50)
IDX=$SLURM_ARRAY_TASK_ID
TURN_IDX=$((IDX % 10))
T=${TURNS[$TURN_IDX]}

CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints

if [ $IDX -lt 10 ]; then
    CKPT=$CKPT_DIR/pattern_simple_direct_H512_wheneven_turn${T}.pt
    LABEL="specialist"
else
    CKPT=$CKPT_DIR/pattern_simple_direct_H512_wheneven.pt
    LABEL="unified"
fi

echo "Probe: model=$LABEL  turn=$T  ckpt=$CKPT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_per_turn_probe.py \
    --ckpt "$CKPT" --hidden 512 --target-turn "$T" --epochs 3
