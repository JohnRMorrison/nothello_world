"""Train MLP on move features using chunk-streaming (constant memory).

Loads one chunk at a time — scales to any number of games.

Usage:
    python train_streaming.py --hidden 1024 --epochs 10 [--features when+even]
"""
import sys, os, json
sys.path.insert(0, '.')

import argparse
import numpy as np
import torch
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _train_mlp_streaming, get_device, N_MOVES,
)

# Spatial feature transform (row/col per square)
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})
_ROWS = np.array([pos // 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)
_COLS = np.array([pos % 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)

def _add_spatial(X):
    """(N, 180) -> (N, 300): append played*row and played*col."""
    played = X[:, :N_MOVES]
    rows_feat = played * torch.from_numpy(_ROWS)
    cols_feat = played * torch.from_numpy(_COLS)
    return torch.cat([X, rows_feat, cols_feat], dim=1)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--features", type=str, default="all",
                    choices=["all", "when+even", "played+when", "played+even",
                             "when", "played", "even"],
                    help="Which feature subset to use")
parser.add_argument("--spatial", action="store_true",
                    help="Add 120-d spatial features (row/col per square) -> 300-d")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

# Feature column selections
FEATURE_SUBSETS = {
    "all":          (list(range(3 * N_MOVES)), 3 * N_MOVES),
    "when+even":    (list(range(N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "played+when":  (list(range(0, 2 * N_MOVES)), 2 * N_MOVES),
    "played+even":  (list(range(0, N_MOVES)) + list(range(2 * N_MOVES, 3 * N_MOVES)), 2 * N_MOVES),
    "when":         (list(range(N_MOVES, 2 * N_MOVES)), N_MOVES),
    "played":       (list(range(0, N_MOVES)), N_MOVES),
    "even":         (list(range(2 * N_MOVES, 3 * N_MOVES)), N_MOVES),
}

feature_cols, input_dim = FEATURE_SUBSETS[args.features]
if args.features == "all":
    feature_cols = None  # no slicing needed
    input_dim = 3 * N_MOVES

transform_fn = None
if args.spatial:
    if args.features != "all":
        print("ERROR: --spatial requires --features all (needs played[] columns)")
        sys.exit(1)
    transform_fn = _add_spatial
    input_dim = 300  # 180 + 120

device = get_device()
feat_desc = f"{input_dim}-d" + (" +spatial" if args.spatial else "")
print(f"Device: {device}")
print(f"Features: {args.features} ({feat_desc}), H={args.hidden}, {args.epochs} epochs")

chunk_dir = os.path.join(args.output_dir, "feature_chunks")

save_dir = os.path.join(args.output_dir, "mlp_checkpoints")
os.makedirs(save_dir, exist_ok=True)
suffix = "_spatial" if args.spatial else ""
save_path = os.path.join(save_dir, f"mlp_{args.features}{suffix}_H{args.hidden}_streaming.pt")

acc = _train_mlp_streaming(
    chunk_dir, device, input_dim, args.hidden,
    feature_cols=feature_cols,
    transform_fn=transform_fn,
    epochs=args.epochs,
    save_path=save_path,
)

print(f"\nFinal: {args.features}{suffix} H={args.hidden}: {acc:.4%}")
