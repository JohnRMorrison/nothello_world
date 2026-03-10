"""Ablation: which of the 3 feature groups (played/when/even) matter?

Trains H=1024 MLP with different feature subsets:
  - 60-d: when only
  - 120-d: played + when
  - 120-d: when + even
  - 180-d: all three (baseline)

Usage:
    python feature_ablation.py [--max-games 100000] [--epochs 10]
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    load_games, _build_move_features_batch, _train_mlp_nanda,
    get_device, POS_START, POS_END, N_MOVES,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max-games", type=int, default=100000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--hidden", type=int, default=1024)
args = parser.parse_args()

device = get_device()
print(f"Device: {device}")

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
print(f"  Train: {tr_X.shape}, Eval: {ev_X.shape}")

# Feature slices: played=0:60, when=60:120, even=120:180
subsets = {
    "when_only (60-d)":        list(range(N_MOVES, 2*N_MOVES)),
    "played+when (120-d)":     list(range(0, 2*N_MOVES)),
    "when+even (120-d)":       list(range(N_MOVES, 3*N_MOVES)),
    "played+even (120-d)":     list(range(0, N_MOVES)) + list(range(2*N_MOVES, 3*N_MOVES)),
    "played_only (60-d)":      list(range(0, N_MOVES)),
    "even_only (60-d)":        list(range(2*N_MOVES, 3*N_MOVES)),
    "all (180-d)":             list(range(3*N_MOVES)),
}

results = {}
for name, cols in subsets.items():
    input_dim = len(cols)
    print(f"\n{'='*60}")
    print(f"  {name}: {input_dim} features, H={args.hidden}")
    print(f"{'='*60}")

    tr_sub = tr_X[:, cols]
    ev_sub = ev_X[:, cols]

    acc = _train_mlp_nanda(
        tr_sub, tr_Y, tr_pos, ev_sub, ev_Y, ev_pos,
        device, input_dim, args.hidden, epochs=args.epochs)
    results[name] = acc
    print(f"  -> {name}: {acc:.4%}")

print(f"\n{'='*60}")
print("FEATURE ABLATION RESULTS")
print(f"{'='*60}")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {acc:.4%}")
