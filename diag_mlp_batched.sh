#!/bin/bash
#SBATCH --job-name=diag_mlp
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/diag_mlp_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

CKPT=${CKPT:-experiments/mathematical_transformation_experiments/heuristic_probe_results/pattern_detector_checkpoints/pattern_simple_direct_H512_playedeven_seed44.pt}
CHUNK=${CHUNK:-experiments/mathematical_transformation_experiments/heuristic_probe_results/feature_chunks/chunk_ext_0039.npz}
N=${N:-5}

python diag_mlp_batched.py --ckpt $CKPT --chunk $CHUNK --num-positions $N
