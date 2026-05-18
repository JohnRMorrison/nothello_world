#!/bin/bash
# Train H=4096 movegrid pattern detector on extended-range chunks (turns 5-58).
# Saves to pattern_simple_direct_H4096_move_grid_ext.pt.

#SBATCH --job-name=pat_h4kx
#SBATCH -c 4
#SBATCH --time=18:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pat_h4kx_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

# Back up the existing (turns-5-53) checkpoint so the new extended-range
# training doesn't clobber it. train_pattern_simple.py saves to
# pattern_simple_direct_H4096_move_grid.pt; we'll rename after training.
CKPT_DIR=experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints
ORIG="$CKPT_DIR/pattern_simple_direct_H4096_move_grid.pt"
BACKUP="$CKPT_DIR/pattern_simple_direct_H4096_move_grid_orig5_53.pt"

if [ -f "$ORIG" ] && [ ! -f "$BACKUP" ]; then
    cp "$ORIG" "$BACKUP"
    echo "Backed up existing ckpt to $BACKUP"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_pattern_simple.py \
    --hidden 4096 --mode direct --epochs 3 --features move_grid \
    --loss bce --chunk-prefix chunk_ext_

# Rename the freshly-trained extended-range checkpoint
mv "$ORIG" "$CKPT_DIR/pattern_simple_direct_H4096_move_grid_ext.pt"
# Restore the original at the standard path so other scripts find it
cp "$BACKUP" "$ORIG"
echo "Saved extended ckpt; restored original at $ORIG"
