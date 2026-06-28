"""Diagnostic: compare batched mlp_scores_batch to single-position mlp_cell_scores.

Loads a few real positions from a chunk, computes scores through both
pipelines, prints the per-cell scores side by side.  If they differ, the bug
is in the batched version.

Usage:
    python diag_mlp_batched.py \\
      --ckpt experiments/.../pattern_simple_direct_H512_playedeven_seed44.pt \\
      --chunk experiments/.../feature_chunks/chunk_ext_0039.npz \\
      --num-positions 5
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from compare_v4_vs_mlp import load_mlp, mlp_cell_scores, C64_TO_C60, C60_TO_C64
from train_aggregator_readout import mlp_scores_batch, slice_played_even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunk', required=True)
    ap.add_argument('--num-positions', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    mlp = load_mlp(args.ckpt, args.hidden, device)

    # Load a few rows
    with np.load(args.chunk) as z:
        N = z['features'].shape[0]
        sample = np.random.RandomState(0).choice(
            N, size=args.num_positions, replace=False
        )
        sample.sort()
        feats_180 = z['features'][sample].astype(np.float32)
        positions = z['positions'][sample].astype(np.int64)

    feats_120 = slice_played_even(feats_180)

    # Batched version
    x_dev = torch.from_numpy(feats_120).to(device)
    pos_dev = torch.from_numpy(positions).to(device)
    batched_scores = mlp_scores_batch(mlp, x_dev, pos_dev, device).cpu().numpy()

    # For each position, also compute via the single-position path.
    # The single-position path expects a `game` (list of cell-64 indices) and `k`
    # (number of moves played).  Reconstruct game from the chunk's "played" feature.
    # Note: this reconstruction can be ambiguous because chunk features don't
    # record the play order (the 'when' column does in 180-d).  We use 'when'
    # to recover order.
    # features layout: [played(60), when(60), even(60)]
    print(f"\n{'pos':>4}  {'k':>3}  {'argmax':>7}  {'batched[argmax]':>16}  "
          f"{'single[argmax]':>15}  {'match?':>8}")
    print("-" * 72)
    for i in range(args.num_positions):
        played = feats_180[i, :60]
        when = feats_180[i, 60:120]
        # Reconstruct play order from when (step = when*60 - 1)
        played_idx = np.where(played > 0.5)[0]
        steps = (when[played_idx] * 60 - 1).round().astype(int)
        order = np.argsort(steps)
        game_60 = played_idx[order]  # 60-cell indices in play order
        game_64 = [C60_TO_C64.get(int(c)) for c in game_60]
        game_64 = [c for c in game_64 if c is not None]
        k = len(game_64)
        # Sanity: k should equal position+something
        # In fired_patterns / chunk_ext convention: position = t, features = state at step t-1.
        # Actually let's just check
        single_scores = mlp_cell_scores(mlp, game_64 + [0], k, device)
        # ^ pads game to len k+1 since mlp_cell_scores uses game[:k]
        # Actually mlp_cell_scores uses game[:k], so game just needs to have at least k items
        argmax_batched = int(batched_scores[i].argmax())
        argmax_single = int(single_scores.argmax())
        diff_at_argmax = abs(batched_scores[i, argmax_batched]
                              - single_scores[argmax_batched])
        match = '✓' if argmax_batched == argmax_single else '✗ MISMATCH'
        print(f"  {positions[i]:>4}  {k:>3}  {argmax_batched:>7}  "
              f"{batched_scores[i, argmax_batched]:>16.4f}  "
              f"{single_scores[argmax_batched]:>15.4f}  {match:>8}")

    # Detailed comparison for position 0
    print("\nDetailed comparison for position 0:")
    print(f"  batched (top-10 cells): "
          f"{np.argsort(-batched_scores[0])[:10].tolist()}")
    # Recompute single for position 0
    played = feats_180[0, :60]
    when = feats_180[0, 60:120]
    played_idx = np.where(played > 0.5)[0]
    steps = (when[played_idx] * 60 - 1).round().astype(int)
    order = np.argsort(steps)
    game_60 = played_idx[order]
    game_64 = [C60_TO_C64.get(int(c)) for c in game_60]
    game_64 = [c for c in game_64 if c is not None]
    k = len(game_64)
    print(f"  k={k}  position={positions[0]}")
    single_scores = mlp_cell_scores(mlp, game_64 + [0], k, device)
    print(f"  single  (top-10 cells): {np.argsort(-single_scores)[:10].tolist()}")
    print(f"  max abs diff: {np.abs(batched_scores[0] - single_scores).max():.6f}")


if __name__ == '__main__':
    main()
