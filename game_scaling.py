"""Game scaling analysis: accuracy vs number of training games.

Loads raw games (NOT precomputed chunks) and splits by game to ensure
proper train/eval separation. Builds 180-d features from scratch.

Usage:
    python game_scaling.py --n-games 5000   [--epochs 10] [--hidden 1024]
    python game_scaling.py --n-games 10000
    sbatch --array=0-5 game_scaling.sh
"""
import sys, os, json
sys.path.insert(0, '.')

import torch
import numpy as np
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    load_games, _build_move_features_batch, _train_mlp_nanda,
    get_device, POS_START, POS_END, N_MOVES, LENGTH,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n-games", type=int, required=True,
                    help="Number of games to use")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--max-files", type=int, default=None,
                    help="Max game files to load (default: all)")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

device = get_device()
print(f"Device: {device}")

# Load games and subsample
print(f"Loading games...")
all_games = load_games(max_files=args.max_files)
print(f"  Total available: {len(all_games)} games")

if args.n_games > len(all_games):
    print(f"  WARNING: requested {args.n_games} but only {len(all_games)} available")
    args.n_games = len(all_games)

games = all_games[:args.n_games]
del all_games

# Split by GAME (not by sample) — proper train/eval separation
n_eval_games = max(int(len(games) * 0.1), 100)
train_games = games[:len(games) - n_eval_games]
eval_games = games[len(games) - n_eval_games:]
print(f"  Using {len(games)} games: {len(train_games)} train, {len(eval_games)} eval")

# Build 180-d features
print("Building train features (180-d)...")
tr_X, tr_Y, tr_pos = _build_move_features_batch(
    train_games, POS_START, POS_END, include_pairwise=False)
del train_games

print("Building eval features (180-d)...")
ev_X, ev_Y, ev_pos = _build_move_features_batch(
    eval_games, POS_START, POS_END, include_pairwise=False)
del eval_games, games

print(f"  Train: {tr_X.shape} ({tr_X.shape[0]} samples)")
print(f"  Eval:  {ev_X.shape} ({ev_X.shape[0]} samples)")

# Train
H = args.hidden
print(f"\n{'='*60}")
print(f"  180-d features, H={H}, {args.n_games} games, {args.epochs} epochs")
print(f"{'='*60}")

acc = _train_mlp_nanda(
    tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
    device, 180, H, epochs=args.epochs)
if isinstance(acc, tuple):
    acc = acc[0]

print(f"\n  -> {args.n_games} games, H={H}: {acc:.4%}")

# Save results (append to existing)
out_path = os.path.join(args.output_dir, "game_scaling_results.json")
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
else:
    results = {}
results[f"{args.n_games}_games_H{H}"] = float(acc)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
