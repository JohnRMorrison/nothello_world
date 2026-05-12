#!/bin/bash
#SBATCH --job-name=rf_poc
#SBATCH -c 4
#SBATCH --time=01:30:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rf_poc_%A_%a.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p logs
cd $SLURM_SUBMIT_DIR

# Array task 0 = raw counts; 1 = mod 2
if [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
    EXTRA="--use-mod2"
    LBL="mod2"
else
    EXTRA=""
    LBL="raw"
fi

echo "Rule-firings POC: $LBL encoding"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python rule_firings_poc.py \
    --n-games 20000 --max-files 4 --hidden 512 --epochs 20 $EXTRA
