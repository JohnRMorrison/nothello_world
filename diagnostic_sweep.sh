#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=nklab,burst
#SBATCH --exclude=ax01,ax02,ax03,ax04,ax05,ax06,ax07,ax09
#SBATCH --output=logs/diag_sweep_%A_%a.out
#SBATCH --error=logs/diag_sweep_%A_%a.err
#SBATCH --job-name=diag_sw
#SBATCH --array=0-10

source activate othello

python3 << 'PYEOF'
import pickle, torch, sys, numpy as np, json, os, gc
sys.path.insert(0, '.')
from behavioral_utils import load_model
from mingpt.dataset import CharDataset
from finetune_corruption import evaluate, build_legal_mask
from data.othello import OthelloBoardState
from hand_crafted_flanking import enumerate_flanking_patterns
from sensitivity_param_search import (
    select_rules_for_group, apply_corruption,
    precompute_pattern_arrays_extended, generate_games_extended
)
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy

task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# Conditions:
# 0-3: n_rules sweep (2,4,6,8 random rules, flip_color, bs=16, lr=5e-5)
# 4-6: batch_size sweep (bs=32,64,128 on left_board 100 rules, lr=5e-5)
# 7-10: lr=3e-4 sweep (5,10,50,100 random rules, flip_color, bs=16)
configs = [
    {'n_rules': 2,   'bs': 16,  'lr': 5e-5, 'group': 'random'},
    {'n_rules': 4,   'bs': 16,  'lr': 5e-5, 'group': 'random'},
    {'n_rules': 6,   'bs': 16,  'lr': 5e-5, 'group': 'random'},
    {'n_rules': 8,   'bs': 16,  'lr': 5e-5, 'group': 'random'},
    {'n_rules': 100, 'bs': 32,  'lr': 5e-5, 'group': 'left_board'},
    {'n_rules': 100, 'bs': 64,  'lr': 5e-5, 'group': 'left_board'},
    {'n_rules': 100, 'bs': 128, 'lr': 5e-5, 'group': 'left_board'},
    {'n_rules': 5,   'bs': 16,  'lr': 3e-4, 'group': 'random'},
    {'n_rules': 10,  'bs': 16,  'lr': 3e-4, 'group': 'random'},
    {'n_rules': 50,  'bs': 16,  'lr': 3e-4, 'group': 'random'},
    {'n_rules': 100, 'bs': 16,  'lr': 3e-4, 'group': 'random'},
]

cfg = configs[task_id]
label = f"nrules={cfg['n_rules']}_bs={cfg['bs']}_lr={cfg['lr']}_{cfg['group']}"
print(f"Task {task_id}: {label}", flush=True)
print(f"Started at: {__import__('time').strftime('%c')}", flush=True)

model_orig, dataset, device = load_model()

# Build standard test set
print("Building standard test set...", flush=True)
all_std_games = pickle.load(open('data/othello_synthetic/gen10e5__20220324_153929.pickle', 'rb'))
std_test_games = all_std_games[:2000]
std_legal = []
for g in std_test_games:
    board = OthelloBoardState()
    lm = []
    for move in g:
        lm.append(board.get_valid_moves())
        board.umpire(move)
    std_legal.append(lm)
std_ds = CharDataset(std_test_games)
std_mask = build_legal_mask(std_test_games, std_legal, std_ds.stoi, std_ds.block_size, std_ds.vocab_size)
std_loader = DataLoader(std_ds, batch_size=64, shuffle=False)
del all_std_games
gc.collect()

# Select rules
patterns = enumerate_flanking_patterns()
sens_data = json.load(open('behavioral_data/sensitivity.json'))
rng = np.random.RandomState(42)

if cfg['group'] == 'random':
    rule_ids = rng.choice(960, cfg['n_rules'], replace=False).tolist()
else:
    rule_ids = select_rules_for_group(cfg['group'], sens_data, rng)

print(f"Selected {len(rule_ids)} rules", flush=True)

# Apply flip_color corruption
corrupted_patterns, n_modified = apply_corruption('flip_color', deepcopy(patterns), rule_ids, rng)
print(f"Modified {n_modified} rules", flush=True)

# Generate or load games
games_dir = f"experiments/diag_sweep/task_{task_id:02d}"
os.makedirs(games_dir, exist_ok=True)
games_path = os.path.join(games_dir, 'train_games.pickle')
legal_path = os.path.join(games_dir, 'train_legal.pickle')

if os.path.exists(games_path):
    print(f"Loading existing games...", flush=True)
    with open(games_path, 'rb') as f:
        train_games = pickle.load(f)
    with open(legal_path, 'rb') as f:
        train_legal = pickle.load(f)
else:
    print(f"Generating 200000 games...", flush=True)
    arrays = precompute_pattern_arrays_extended(corrupted_patterns)
    train_games, train_legal = generate_games_extended(
        arrays, num_games=200000, rng=np.random.RandomState(42), save_legal=True)
    # Filter short games
    filtered_games = []
    filtered_legal = []
    for g, l in zip(train_games, train_legal):
        if len(g) >= 5:
            filtered_games.append(g)
            filtered_legal.append(l)
    train_games = filtered_games
    train_legal = filtered_legal
    with open(games_path, 'wb') as f:
        pickle.dump(train_games, f)
    with open(legal_path, 'wb') as f:
        pickle.dump(train_legal, f)

n_train = min(190000, len(train_games))
print(f"Training on {n_train} games, bs={cfg['bs']}, lr=5e-5", flush=True)

# Build corrupted test set
cor_test_games = train_games[n_train:n_train+10000]
cor_test_legal = train_legal[n_train:n_train+10000]
cor_ds = CharDataset(cor_test_games)
cor_mask = build_legal_mask(cor_test_games, cor_test_legal, cor_ds.stoi, cor_ds.block_size, cor_ds.vocab_size)
cor_loader = DataLoader(cor_ds, batch_size=64, shuffle=False)

# Train
model = deepcopy(model_orig)
model.train()
optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

train_ds = CharDataset(train_games[:n_train])
train_loader = DataLoader(train_ds, batch_size=cfg['bs'], shuffle=True)

total_steps = len(train_loader)
eval_at = set([0, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000,
               min(10000, total_steps - 1), total_steps - 1])

results = {'label': label, 'config': cfg, 'eval_steps': [],
           'std_acc': [], 'std_lpm': [], 'cor_acc': [], 'cor_lpm': []}

for step, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step in eval_at:
        model.eval()
        sl, sa, sr, slpm = evaluate(model, std_loader, device, std_mask)
        cl, ca, cr, clpm = evaluate(model, cor_loader, device, cor_mask)
        print(f"  Step {step}: std_acc={sa:.4f} std_lpm={slpm:.4f} "
              f"cor_acc={ca:.4f} cor_lpm={clpm:.4f}", flush=True)
        results['eval_steps'].append(step)
        results['std_acc'].append(float(sa))
        results['std_lpm'].append(float(slpm))
        results['cor_acc'].append(float(ca))
        results['cor_lpm'].append(float(clpm))
        model.train()

# Save results
out_path = f"experiments/diag_sweep/task_{task_id:02d}.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_path}", flush=True)
print(f"Finished at: {__import__('time').strftime('%c')}", flush=True)
PYEOF
