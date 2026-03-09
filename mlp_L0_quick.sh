#!/bin/bash
#SBATCH --job-name=mlp_L0
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mlp_L0_%j.out
#SBATCH --account=nklab
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09

module load cuda/11.8.0
source /share/apps/anaconda3-2019.03/etc/profile.d/conda.sh
conda activate othello
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SLURM_SUBMIT_DIR

python -u -c "
import sys; sys.path.insert(0, '.')
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    load_model, load_games, collect_activations_and_labels,
    _train_mlp_nanda, get_device, POS_START, LENGTH
)
device = get_device()
print(f'Device: {device}', flush=True)
model, bs = load_model('ckpts/gpt_synthetic.ckpt', device)
print('Model loaded', flush=True)
games = load_games()[:250000]
print(f'Loaded {len(games)} games', flush=True)
tr, ev = games[:225000], games[225000:]
print('Extracting train activations...', flush=True)
tr_X, tr_Y = collect_activations_and_labels(model, tr, device, 0, bs)
tr_X, tr_Y = tr_X.cpu(), tr_Y.cpu()
tr_pos = torch.tensor([POS_START + (i % LENGTH) for i in range(len(tr_X))])
print(f'Train: {tr_X.shape}', flush=True)
print('Extracting eval activations...', flush=True)
ev_X, ev_Y = collect_activations_and_labels(model, ev, device, 0, bs)
ev_X, ev_Y = ev_X.cpu(), ev_Y.cpu()
ev_pos = torch.tensor([POS_START + (i % LENGTH) for i in range(len(ev_X))])
print(f'Eval: {ev_X.shape}', flush=True)
# Free model from GPU
del model
torch.cuda.empty_cache()
import gc; gc.collect()
print('Training MLP H=1024...', flush=True)
acc = _train_mlp_nanda(tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos, device, 512, 1024, epochs=16)
print(f'MLP H=1024 on L0: {acc}')
"
