"""Train MLP on 300-d features: 180-d base + 120-d spatial (row/col per square).

The spatial features are computed on-the-fly from the played[] features
and fixed row/col coordinates. No need to recompute chunks.

Usage:
    python spatial_features.py --hidden 1024 [--precomputed]
    python spatial_features.py --hidden 2048 [--precomputed]
"""
import sys, os, json
sys.path.insert(0, '.')

import torch
import numpy as np
from experiments.mathematical_transformation_experiments.heuristic_probe_experiments import (
    _load_features, _train_mlp_nanda, get_device, N_MOVES, LENGTH,
)

# Valid moves: 0-63 excluding center squares 27,28,35,36
_VALID_MOVES = sorted(set(range(64)) - {27, 28, 35, 36})

# Precompute row/col for each of the 60 valid squares (normalized to [0,1])
_ROWS = np.array([pos // 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)  # (60,)
_COLS = np.array([pos % 8 / 7.0 for pos in _VALID_MOVES], dtype=np.float32)   # (60,)


def add_spatial_features(X):
    """Append 120-d spatial features to 180-d base features.

    For each square i (0-59):
      spatial_row[i] = played[i] * row(i) / 7
      spatial_col[i] = played[i] * col(i) / 7

    Input: (N, 180) -> Output: (N, 300)
    """
    played = X[:, :N_MOVES]  # (N, 60)
    rows_feat = played * torch.from_numpy(_ROWS)  # (N, 60)
    cols_feat = played * torch.from_numpy(_COLS)   # (N, 60)
    return torch.cat([X, rows_feat, cols_feat], dim=1)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max-games", type=int, default=1000000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--hidden", type=int, nargs='+', default=[1024, 2048])
parser.add_argument("--precomputed", action="store_true")
parser.add_argument("--output-dir",
                    default="experiments/mathematical_transformation_experiments/heuristic_probe_results")
args = parser.parse_args()

device = get_device()
print(f"Device: {device}")

# Load data
max_samples = args.max_games * LENGTH if args.max_games else None
chunk_dir = os.path.join(args.output_dir, "feature_chunks")
chunk_files = sorted(os.path.join(chunk_dir, f)
                     for f in os.listdir(chunk_dir) if f.endswith(".npz"))

all_X, all_Y, all_pos = [], [], []
total = 0
for path in chunk_files:
    data = np.load(path)
    n = data['features'].shape[0]
    needed = min(n, max_samples - total) if max_samples else n
    if needed <= 0:
        break
    feat = data['features'][:needed].astype(np.float32)
    # Compute spatial features from played[] columns before converting to tensor
    played = feat[:, :N_MOVES]  # (N, 60)
    rows_feat = played * _ROWS  # (N, 60)
    cols_feat = played * _COLS  # (N, 60)
    feat_300 = np.concatenate([feat, rows_feat, cols_feat], axis=1)  # (N, 300)
    all_X.append(torch.from_numpy(feat_300))
    del feat, played, rows_feat, cols_feat, feat_300
    all_Y.append(torch.from_numpy(data['labels'][:needed].astype(np.int64)))
    all_pos.append(torch.from_numpy(data['positions'][:needed].astype(np.int64)))
    total += needed
    print(f"  {os.path.basename(path)}: {needed} samples (total: {total})", flush=True)
    del data
    if max_samples and total >= max_samples:
        break

X = torch.cat(all_X); del all_X
Y = torch.cat(all_Y); del all_Y
pos = torch.cat(all_pos); del all_pos
print(f"  Feature shape: {X.shape}")

n_eval = max(int(len(X) * 0.1), 49 * 100)
n_train = len(X) - n_eval
tr_X, tr_Y, tr_pos = X[:n_train], Y[:n_train], pos[:n_train]
ev_X, ev_Y, ev_pos = X[n_train:], Y[n_train:], pos[n_train:]
del X, Y, pos

input_dim = tr_X.shape[1]  # 300
print(f"  Train: {tr_X.shape}, Eval: {ev_X.shape}")

ckpt_dir = os.path.join(args.output_dir, "mlp_checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

results = {}
for H in args.hidden:
    print(f"\n{'='*60}")
    print(f"  300-d (180 + spatial) features, H={H}")
    print(f"{'='*60}")

    save_path = os.path.join(ckpt_dir, f"mlp_300_H{H}.pt")
    acc = _train_mlp_nanda(
        tr_X, tr_Y, tr_pos, ev_X, ev_Y, ev_pos,
        device, input_dim, H, epochs=args.epochs, save_path=save_path)
    if isinstance(acc, tuple):
        acc = acc[0]
    results[f"mlp_300_H{H}"] = acc
    print(f"  -> 300-d H={H}: {acc:.4%}")

print(f"\n{'='*60}")
print("SPATIAL FEATURE RESULTS")
print(f"{'='*60}")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {acc:.4%}")

# Save
out_path = os.path.join(args.output_dir, "spatial_results.json")
with open(out_path, "w") as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
print(f"Saved to {out_path}")
