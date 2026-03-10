"""Ablation: which of the 3 feature groups (played/when/even) matter?

Trains H=1024 MLP with different feature subsets:
  - 60-d: when only
  - 120-d: played + when
  - 120-d: when + even
  - 180-d: all three (baseline)

Usage:
    python feature_ablation.py --subset-id 0  [--precomputed] [--max-games 1000000]
    sbatch --array=0-6 feature_ablation.sh
"""
import sys, os, json
sys.path.insert(0, '.')

import torch
import numpy as np
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    load_games, _build_move_features_batch, _train_mlp_nanda,
    _load_features, get_device, POS_START, POS_END, N_MOVES, LENGTH,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max-games", type=int, default=1000000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--subset-id", type=int, default=None,
                    help="Which feature subset (0-6). If None, run all sequentially.")
parser.add_argument("--precomputed", action="store_true")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

# Feature subsets indexed by ID
SUBSETS = [
    ("when_only (60-d)",        slice(N_MOVES, 2*N_MOVES)),
    ("played+when (120-d)",     slice(0, 2*N_MOVES)),
    ("when+even (120-d)",       slice(N_MOVES, 3*N_MOVES)),
    ("played+even (120-d)",     None),  # handled specially below
    ("played_only (60-d)",      slice(0, N_MOVES)),
    ("even_only (60-d)",        slice(2*N_MOVES, 3*N_MOVES)),
    ("all (180-d)",             slice(None)),  # no slicing
]

device = get_device()
print(f"Device: {device}")

# Load data — cap at max_games, load only needed columns to save memory
if args.precomputed:
    max_samples = args.max_games * LENGTH if args.max_games else None
    chunk_dir = os.path.join(args.output_dir, "feature_chunks")
    chunk_files = sorted(os.path.join(chunk_dir, f)
                         for f in os.listdir(chunk_dir) if f.endswith(".npz"))

    # Determine which columns we need
    if args.subset_id is not None:
        _, col_spec = SUBSETS[args.subset_id]
    else:
        col_spec = slice(None)  # need all columns for multiple subsets

    all_X, all_Y, all_pos = [], [], []
    total = 0
    for path in chunk_files:
        data = np.load(path)
        n = data['features'].shape[0]
        needed = min(n, max_samples - total) if max_samples else n
        if needed <= 0:
            break
        feat = data['features'][:needed]
        # Slice columns BEFORE converting to float32 to save memory
        if col_spec is not None and col_spec != slice(None):
            feat = feat[:, col_spec]
        elif args.subset_id == 3:  # played+even: non-contiguous
            feat = np.concatenate([feat[:, :N_MOVES], feat[:, 2*N_MOVES:]], axis=1)
        all_X.append(torch.from_numpy(feat.astype(np.float32)))
        all_Y.append(torch.from_numpy(data['labels'][:needed].astype(np.int64)))
        all_pos.append(torch.from_numpy(data['positions'][:needed].astype(np.int64)))
        total += needed
        print(f"  {os.path.basename(path)}: {needed} samples (total: {total})", flush=True)
        del data, feat
        if max_samples and total >= max_samples:
            break

    X = torch.cat(all_X); del all_X
    Y = torch.cat(all_Y); del all_Y
    pos = torch.cat(all_pos); del all_pos

    n_eval = max(int(len(X) * 0.1), 49 * 100)
    n_train = len(X) - n_eval
    tr_X, tr_Y, tr_pos = X[:n_train], Y[:n_train], pos[:n_train]
    ev_X, ev_Y, ev_pos = X[n_train:], Y[n_train:], pos[n_train:]
    del X, Y, pos
else:
    games = load_games()
    if len(games) > args.max_games:
        games = games[:args.max_games]
    n_eval = max(int(len(games) * 0.1), 100)
    train_games = games[:len(games) - n_eval]
    eval_games = games[len(games) - n_eval:]
    print(f"Using {len(games)} games ({len(train_games)} train, {len(eval_games)} eval)")
    print("Building 180-d features (train)...")
    tr_X, tr_Y, tr_pos = _build_move_features_batch(train_games, POS_START, POS_END, include_pairwise=False)
    print("Building 180-d features (eval)...")
    ev_X, ev_Y, ev_pos = _build_move_features_batch(eval_games, POS_START, POS_END, include_pairwise=False)

input_dim = tr_X.shape[1]
print(f"  Train: {tr_X.shape}, Eval: {ev_X.shape}")

# Select which subsets to run
if args.subset_id is not None:
    to_run = [(SUBSETS[args.subset_id][0], None)]  # columns already selected
else:
    to_run = SUBSETS

results = {}
for name, col_spec in to_run:
    if col_spec is not None and col_spec != slice(None):
        # Need to slice (only when running all subsets sequentially)
        if args.subset_id == 3 or (isinstance(col_spec, type(None))):
            cols = list(range(0, N_MOVES)) + list(range(2*N_MOVES, 3*N_MOVES))
            tr_sub = tr_X[:, cols].contiguous()
            ev_sub = ev_X[:, cols].contiguous()
        else:
            tr_sub = tr_X[:, col_spec].contiguous()
            ev_sub = ev_X[:, col_spec].contiguous()
        cur_dim = tr_sub.shape[1]
    else:
        tr_sub = tr_X
        ev_sub = ev_X
        cur_dim = input_dim

    print(f"\n{'='*60}")
    print(f"  {name}: {cur_dim} features, H={args.hidden}")
    print(f"{'='*60}")

    acc = _train_mlp_nanda(
        tr_sub, tr_Y, tr_pos, ev_sub, ev_Y, ev_pos,
        device, cur_dim, args.hidden, epochs=args.epochs)
    results[name] = acc
    print(f"  -> {name}: {acc:.4%}")

    if tr_sub is not tr_X:
        del tr_sub, ev_sub

print(f"\n{'='*60}")
print("FEATURE ABLATION RESULTS")
print(f"{'='*60}")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {acc:.4%}")

# Save results
out_path = os.path.join(args.output_dir, "feature_ablation_results.json")
if os.path.exists(out_path):
    with open(out_path) as f:
        all_results = json.load(f)
else:
    all_results = {}
all_results.update({k: float(v) for k, v in results.items()})
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved to {out_path}")
